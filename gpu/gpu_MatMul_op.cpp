#include <iostream>

#include "absl/container/inlined_vector.h"
#include "gpuBackend.h"
#include "shaders/headers/shaderMatMul.hpp"
#include "shaders/headers/shaderTranspose.hpp"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "vulten_device.h"

namespace vulten_plugin {

static std::vector<uint32_t> spirv_matmul;
static std::vector<uint32_t> spirv_transpose;

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

template <class T>
struct MatMulOp {
  MatMulOp() : transA_(false), transB_(false) {}
  bool transA_, transB_;
};

template <typename T>
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

template <typename T>
void MatMulOp_Delete(void* kernel) {
  if (kernel != nullptr) {
    delete static_cast<MatMulOp<T>*>(kernel);
  }
}

template <typename T>
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
      (outDims[0] * outDims[1]) * sizeof(T), status.get()));
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
                {*a_ptr, transA}, spirv_transpose,
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
                {*b_ptr, transB}, spirv_transpose,
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
      spirv_matmul, kp::Workgroup({uint32_t(outDims[0]), uint32_t(outDims[1])}),
      std::vector<uint32_t>{ax, ay, bx, by}, {});

  stream->instance->mngr->sequence(stream->instance->mainQueue)
      ->record<kp::OpAlgoDispatch>(algo)
      ->eval();
}

template <typename T>
void RegisterMatMulOpKernel(const char* device_type) {
  StatusSafePtr status(TF_NewStatus());
  auto* builder =
      TF_NewKernelBuilder("MatMul", device_type, MatMulOp_Create<T>,
                          &MatMulOp_Compute<T>, &MatMulOp_Delete<T>);
  TF_KernelBuilder_TypeConstraint(builder, "T", TF_FLOAT, status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout << " Error while registering MatMul kernel with attribute T";
  TF_RegisterKernelBuilder("MatMulOp", builder, status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout << " Error while registering MatMul kernel";
}

}  // namespace vulten_plugin

void RegisterDeviceMatMul(const char* device_type) {
  vulten_plugin::spirv_matmul.resize(
      kp::shader_data::___shaders_MatMul_comp_spv_len / 4);
  memcpy(&vulten_plugin::spirv_matmul[0],
         kp::shader_data::___shaders_MatMul_comp_spv,
         kp::shader_data::___shaders_MatMul_comp_spv_len);

  vulten_plugin::spirv_transpose.resize(
      kp::shader_data::___shaders_Transpose_comp_spv_len / 4);
  memcpy(&vulten_plugin::spirv_transpose[0],
         kp::shader_data::___shaders_Transpose_comp_spv,
         kp::shader_data::___shaders_Transpose_comp_spv_len);

  vulten_plugin::RegisterMatMulOpKernel<float>(device_type);
}