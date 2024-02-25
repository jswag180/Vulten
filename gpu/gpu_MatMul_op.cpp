#include "Vulten_backend/ops/matMul/MatMul_op.h"
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

  tensor_utills::Input_tensor a =
      tensor_utills::get_input_tensor("MatMulOp:a", 0, ctx, status.get());
  vulten_ops::mat_mul::Mat_size a_mat_size = vulten_ops::mat_mul::Mat_size();
  a_mat_size.x = a.dims[0];
  a_mat_size.y = a.dims[1];

  tensor_utills::Input_tensor b =
      tensor_utills::get_input_tensor("MatMulOp:b", 1, ctx, status.get());
  vulten_ops::mat_mul::Mat_size b_mat_size = vulten_ops::mat_mul::Mat_size();
  b_mat_size.x = b.dims[0];
  b_mat_size.y = b.dims[1];

  uint32_t ax = matMulOpInfo->transA_ ? a.dims[1] : a.dims[0];
  uint32_t ay = matMulOpInfo->transA_ ? a.dims[0] : a.dims[1];
  uint32_t bx = matMulOpInfo->transB_ ? b.dims[1] : b.dims[0];
  uint32_t by = matMulOpInfo->transB_ ? b.dims[0] : b.dims[1];
  if (ay != bx) {
    std::cerr << "Error: incompatable sized matrixes\n";
    return;
  }

  absl::InlinedVector<int64_t, 4> out_dims = {ax, by};
  tensor_utills::Output_tensor output = tensor_utills::make_output_tensor(
      "MatMulOp:output", 0, out_dims, ctx, status.get());

  SP_Stream stream = TF_GetStream(ctx, status.get());
  vulten_backend::Instance* inst = stream->instance;

  vulten_ops::mat_mul::run_op(inst, (vulten_ops::Data_type)T, a.vulten_tensor,
                              matMulOpInfo->transA_, a_mat_size,
                              b.vulten_tensor, matMulOpInfo->transB_,
                              b_mat_size, output.vulten_tensor);
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
  CALL_HALF(REGISTER_KERNEL)
  CALL_DOUBLE(REGISTER_KERNEL)
  REGISTER_KERNEL(TF_INT32)
  CALL_INT64(REGISTER_KERNEL)
  CALL_COMPLEX(REGISTER_KERNEL)
}