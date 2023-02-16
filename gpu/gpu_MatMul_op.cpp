#include "Vulten_backend/ops/MatMul_op.h"
#include "absl/container/inlined_vector.h"
#include "scope_timer.h"
#include "tensor_utills.h"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "vulten_device.h"

template <TF_DataType T>
struct MatMulOp {
  MatMulOp() : transA_(false), transB_(false) {}
  bool transA_, transB_;
};

template <TF_DataType T>
void* MatMulOp_Create(TF_OpKernelConstruction* ctx) {
  auto kernel = new MatMulOp<T>();

  StatusSafePtr status(TF_NewStatus());

  TF_OpKernelConstruction_GetAttrBool(
      ctx, "transpose_a", (unsigned char*)&kernel->transA_, status.get());
  TF_OpKernelConstruction_GetAttrBool(
      ctx, "transpose_b", (unsigned char*)&kernel->transB_, status.get());

  return kernel;
}

template <TF_DataType T>
void MatMulOp_Delete(void* kernel) {
  if (kernel != nullptr) {
    delete static_cast<MatMulOp<T>*>(kernel);
  }
}

template <TF_DataType T>
void MatMulOp_Compute(void* kernel, TF_OpKernelContext* ctx) {
  SCOPE_TIMER("MatMulOp")

  MatMulOp<T>* matMulOpInfo = static_cast<MatMulOp<T>*>(kernel);

  StatusSafePtr status(TF_NewStatus());

  GET_INPUT_TENSOR("MatMul", a, 0, ctx, status)
  vulten_ops::Mat_size a_mat_size = vulten_ops::Mat_size();
  a_mat_size.x = a_dims[0];
  a_mat_size.y = a_dims[1];

  GET_INPUT_TENSOR("MatMul", b, 1, ctx, status)
  vulten_ops::Mat_size b_mat_size = vulten_ops::Mat_size();
  b_mat_size.x = b_dims[0];
  b_mat_size.y = b_dims[1];

  uint32_t ax = matMulOpInfo->transA_ ? a_dims[1] : a_dims[0];
  uint32_t ay = matMulOpInfo->transA_ ? a_dims[0] : a_dims[1];
  uint32_t bx = matMulOpInfo->transB_ ? b_dims[1] : b_dims[0];
  uint32_t by = matMulOpInfo->transB_ ? b_dims[0] : b_dims[1];
  if (ay != bx) {
    std::cerr << "Error: incompatable sized matrixes\n";
    return;
  }

  absl::InlinedVector<int64_t, 2> out_dims = {ax, by};
  MAKE_OUTPUT_TENSOR("MatMul", output, 0, out_dims, T, ctx, status)

  SP_Stream stream = TF_GetStream(ctx, status.get());
  vulten_backend::Instance* inst = stream->instance;

  vulten_ops::MatMul_op* matMul_op = nullptr;
  std::string op_cache_name = "MatMul";
  inst->main_queue_mutex.lock();
  if (inst->op_chache.find(op_cache_name) == inst->op_chache.end()) {
    inst->op_chache[op_cache_name] =
        (vulten_ops::Vulten_op*)new vulten_ops::MatMul_op(inst);
  }
  matMul_op = (vulten_ops::MatMul_op*)inst->op_chache[op_cache_name];
  inst->main_queue_mutex.unlock();

  matMul_op->run_op((vulten_ops::Data_type)T, a_tensor, matMulOpInfo->transA_,
                    a_mat_size, b_tensor, matMulOpInfo->transB_, b_mat_size,
                    output_tensor);
}

template <TF_DataType T>
void RegisterMatMulOpKernel(const char* device_type) {
  StatusSafePtr status(TF_NewStatus());
  auto* builder =
      TF_NewKernelBuilder("MatMul", device_type, MatMulOp_Create<T>,
                          &MatMulOp_Compute<T>, &MatMulOp_Delete<T>);
  TF_KernelBuilder_TypeConstraint(builder, "T", T, status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout << " Error while registering MatMul kernel with attribute T";
  TF_RegisterKernelBuilder("MatMulOp", builder, status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout << " Error while registering MatMul kernel";
}

void RegisterDeviceMatMul(const char* device_type) {
#define REGISTER_KERNEL(T) RegisterMatMulOpKernel<T>(device_type);

  REGISTER_KERNEL(TF_FLOAT)
  REGISTER_KERNEL(TF_HALF)
  REGISTER_KERNEL(TF_DOUBLE)
  REGISTER_KERNEL(TF_INT32)
  REGISTER_KERNEL(TF_INT64)
  CALL_COMPLEX(REGISTER_KERNEL)
}