#include <iostream>

#include "absl/container/inlined_vector.h"
#include "gpuBackend.h"
#include "shaders/headers/MatMul/MatMul.h"
#include "shaders/headers/Transpose/Transpose.h"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "vulten_device.h"

#define TRANSPOSE 0
#define GENERIC_ALGO 1

namespace vulten_plugin {

struct StatusDeleter {
  void operator()(TF_Status* s) {
    if (s != nullptr) {
      TF_DeleteStatus(s);
    }
  }
};

struct TensorDeleter {
  void operator()(TF_Tensor* t) {
    if (t != nullptr) {
      TF_DeleteTensor(t);
    }
  }
};

using StatusSafePtr = std::unique_ptr<TF_Status, StatusDeleter>;
using TensorSafePtr = std::unique_ptr<TF_Tensor, TensorDeleter>;

template <TF_DataType T>
struct MatMulOp {
  MatMulOp() : transA_(false), transB_(false) {}
  bool transA_, transB_;
};

template <TF_DataType T>
void* MatMulOp_Create(TF_OpKernelConstruction* ctx) {
  auto kernel = new MatMulOp<T>();

  StatusSafePtr status(TF_NewStatus());

  int32_t list_size = 0;
  int32_t total_size = 0;

  unsigned char trans[] = {0, 0};
  TF_OpKernelConstruction_GetAttrBool(ctx, "transpose_a", &trans[0],
                                      status.get());
  TF_OpKernelConstruction_GetAttrBool(ctx, "transpose_b", &trans[1],
                                      status.get());

  kernel->transA_ = trans[0];
  kernel->transB_ = trans[1];

  return kernel;
}

template <TF_DataType T>
void MatMulOp_Delete(void* kernel) {
  if (kernel != nullptr) {
    delete static_cast<MatMulOp<T>*>(kernel);
  }
}

template <TF_DataType T, const std::vector<uint32_t>* spirv[]>
void MatMulOp_Compute(void* kernel, TF_OpKernelContext* ctx) {
  MatMulOp<T>* matMulOp = static_cast<MatMulOp<T>*>(kernel);

  StatusSafePtr status(TF_NewStatus());

  TF_Tensor* inputA = nullptr;
  TF_GetInput(ctx, 0, &inputA, status.get());
  TensorSafePtr inputA_safe_ptr(inputA);
  if (TF_GetCode(status.get()) != TF_OK) {
    TF_OpKernelContext_Failure(ctx, status.get());
    return;
  }
  if (TF_TensorElementCount(inputA_safe_ptr.get()) == 0) return;
  if (TF_NumDims(inputA_safe_ptr.get()) != 2) {
    std::cerr << "Error: input A has more or less then 2 dims\n";
    return;
  }
  absl::InlinedVector<int64_t, 4> aDims(TF_NumDims(inputA_safe_ptr.get()));
  for (auto i = 0; i < TF_NumDims(inputA_safe_ptr.get()); ++i) {
    aDims[i] = TF_Dim(inputA_safe_ptr.get(), i);
  }

  TF_Tensor* inputB = nullptr;
  TF_GetInput(ctx, 1, &inputB, status.get());
  TensorSafePtr inputB_safe_ptr(inputB);
  if (TF_GetCode(status.get()) != TF_OK) {
    TF_OpKernelContext_Failure(ctx, status.get());
    return;
  }
  if (TF_TensorElementCount(inputB_safe_ptr.get()) == 0) return;
  if (TF_NumDims(inputB_safe_ptr.get()) != 2) {
    std::cerr << "Error: input B has more or less then 2 dims\n";
    return;
  }
  absl::InlinedVector<int64_t, 4> bDims(TF_NumDims(inputB_safe_ptr.get()));
  for (auto i = 0; i < TF_NumDims(inputB_safe_ptr.get()); ++i) {
    bDims[i] = TF_Dim(inputB_safe_ptr.get(), i);
  }

  uint32_t ax = matMulOp->transA_ ? aDims[1] : aDims[0];
  uint32_t ay = matMulOp->transA_ ? aDims[0] : aDims[1];
  uint32_t bx = matMulOp->transB_ ? bDims[1] : bDims[0];
  uint32_t by = matMulOp->transB_ ? bDims[0] : bDims[1];
  if (ay != bx) {
    std::cerr << "Error: incompatable sized matrixes\n";
    return;
  }

  int64_t outDims[] = {ax, by};

  TensorSafePtr output_safe_ptr(TF_AllocateOutput(
      ctx, 0, TF_ExpectedOutputDataType(ctx, 0), &outDims[0], 2,
      (outDims[0] * outDims[1]) * TF_DataTypeSize(T), status.get()));
  if (TF_GetCode(status.get()) != TF_OK) {
    TF_OpKernelContext_Failure(ctx, status.get());
    return;
  }

  auto a_ptr = static_cast<std::shared_ptr<kp::TensorT<float>>*>(
      TF_TensorData(inputA_safe_ptr.get()));
  auto b_ptr = static_cast<std::shared_ptr<kp::TensorT<float>>*>(
      TF_TensorData(inputB_safe_ptr.get()));
  auto out_ptr = static_cast<std::shared_ptr<kp::TensorT<float>>*>(
      TF_TensorData(output_safe_ptr.get()));

  SP_Stream stream = TF_GetStream(ctx, status.get());
  MutexScopeLock guard = MutexScopeLock(&stream->instance->mainQueueMutex);

  std::shared_ptr<kp::TensorT<float>> transA;
  std::shared_ptr<kp::Sequence> transASeq;
  if (matMulOp->transA_) {
    std::vector<float> transVec(ax * ay);
    transA = stream->instance->mngr->tensorT<float>(
        {transVec}, kp::Tensor::TensorTypes::eDevice);

    transASeq =
        stream->instance->mngr->sequence(stream->instance->mainQueue)
            ->record<kp::OpAlgoDispatch>(stream->instance->mngr->algorithm(
                {*a_ptr, transA}, *spirv[TRANSPOSE],
                kp::Workgroup({uint32_t(transVec.size())}),
                std::vector<uint32_t>{uint32_t(aDims[0]), uint32_t(aDims[1])},
                {}))
            ->evalAsync();
  }

  std::shared_ptr<kp::TensorT<float>> transB;
  std::shared_ptr<kp::Sequence> transBSeq;
  if (matMulOp->transB_) {
    std::vector<float> transVec(bx * by);
    transB = stream->instance->mngr->tensorT<float>(
        {transVec}, kp::Tensor::TensorTypes::eDevice);

    transBSeq =
        stream->instance->mngr->sequence(stream->instance->mainQueue)
            ->record<kp::OpAlgoDispatch>(stream->instance->mngr->algorithm(
                {*b_ptr, transB}, *spirv[TRANSPOSE],
                kp::Workgroup({uint32_t(transVec.size())}),
                std::vector<uint32_t>{uint32_t(bDims[0]), uint32_t(bDims[1])},
                {}))
            ->evalAsync();
  }

  if (matMulOp->transA_) {
    transASeq->evalAwait();
  }
  if (matMulOp->transB_) {
    transBSeq->evalAwait();
  }

  std::shared_ptr<kp::Algorithm> algo = stream->instance->mngr->algorithm(
      {matMulOp->transA_ ? transA : *a_ptr, matMulOp->transB_ ? transB : *b_ptr,
       *out_ptr},
      *spirv[GENERIC_ALGO],
      kp::Workgroup({uint32_t(outDims[0]), uint32_t(outDims[1])}),
      std::vector<uint32_t>{ax, ay, bx, by}, {});

  stream->instance->mngr->sequence(stream->instance->mainQueue)
      ->record<kp::OpAlgoDispatch>(algo)
      ->eval();
}

template <TF_DataType T, const std::vector<uint32_t>* spirv[]>
void RegisterMatMulOpKernel(const char* device_type) {
  StatusSafePtr status(TF_NewStatus());
  auto* builder =
      TF_NewKernelBuilder("MatMul", device_type, MatMulOp_Create<T>,
                          &MatMulOp_Compute<T, spirv>, &MatMulOp_Delete<T>);
  TF_KernelBuilder_TypeConstraint(builder, "T", T, status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout << " Error while registering MatMul kernel with attribute T";
  TF_RegisterKernelBuilder("MatMulOp", builder, status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout << " Error while registering MatMul kernel";
}

}  // namespace vulten_plugin

void RegisterDeviceMatMul(const char* device_type) {
// The array of pointers is being able in the future implement other matmul
// alogs Format for spirv pointers is 0 is transpose and 1 is generic
#define REGISTER_KERNEL(T, S)                                                 \
  static const std::vector<uint32_t>* spirv_##S[2] = {&shader::Transpose_##S, \
                                                      &shader::MatMul_##S};   \
  vulten_plugin::RegisterMatMulOpKernel<T, spirv_##S>(device_type);

#ifdef MATMUL_FLOAT
  REGISTER_KERNEL(TF_FLOAT, float)
#endif
#ifdef MATMUL_INT
  REGISTER_KERNEL(TF_INT32, int)
#endif
#ifdef MATMUL_UINT
  REGISTER_KERNEL(TF_UINT32, uint)
#endif
#ifdef MATMUL_INT64_T
  REGISTER_KERNEL(TF_INT64, int64_t)
#endif
#ifdef MATMUL_UINT64_T
  REGISTER_KERNEL(TF_UINT64, uint64_t)
#endif
#ifdef MATMUL_INT8_T
  REGISTER_KERNEL(TF_INT8, int8_t)
#endif
#ifdef MATMUL_UINT8_T
  REGISTER_KERNEL(TF_UINT8, uint8_t)
#endif
#ifdef MATMUL_DOUBLE
  REGISTER_KERNEL(TF_DOUBLE, double)
#endif
}