#include "Vulten_backend/ops/basic/Basic_ops.h"
#include "absl/container/inlined_vector.h"
#include "scope_timer.h"
#include "tensor_utills.h"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_tensor.h"
#include "vulten_device.h"

template <TF_DataType T, uint32_t OP>
void BasicOps_Compute(void* kernel, TF_OpKernelContext* ctx) {
  SCOPE_TIMER("BasicOp " + vulten_ops::basic::op_as_str(OP))

  StatusSafePtr status(TF_NewStatus());

  tensor_utills::Input_tensor x =
      tensor_utills::get_input_tensor("BasicOp:x", 0, ctx, status.get());
  if (x.is_empty) {
    return;
  }

  tensor_utills::Input_tensor y =
      tensor_utills::get_input_tensor("BasicOp:y", 1, ctx, status.get());
  if (y.is_empty) {
    return;
  }

  if (x.dims.size() != y.dims.size()) {
    if (x.dims.size() > y.dims.size()) {
      y.dims.resize(x.dims.size(), 1);
    } else {
      x.dims.resize(y.dims.size(), 1);
    }
  }

  absl::InlinedVector<int64_t, 4> res_dims =
      absl::InlinedVector<int64_t, 4>(x.dims.size(), 0);
  for (int64_t i = 0; i < res_dims.size(); i++) {
    res_dims[i] = std::max(x.dims[i], y.dims[i]);
  }

  tensor_utills::Output_tensor output = tensor_utills::make_output_tensor(
      "BasicOp:output", 0, res_dims, ctx, status.get());

  if (output.is_scalar) {
    res_dims.resize(1, 1);
    output.vulten_tensor.num_dims = 1;
  }

  if (x.is_scalar) {
    x.dims.resize(1, 1);
    x.vulten_tensor.num_dims = 1;
  }

  if (y.is_scalar) {
    y.dims.resize(1, 1);
    y.vulten_tensor.num_dims = 1;
  }

  SP_Stream stream = TF_GetStream(ctx, status.get());
  vulten_backend::Instance* inst = stream->instance;

  vulten_ops::basic::run_op(inst, (vulten_ops::Data_type)T, OP, x.vulten_tensor,
                            y.vulten_tensor, output.vulten_tensor);
}

template <TF_DataType T, uint32_t OP>
void RegisterBasicOpKernels(const char* device_type) {
  std::string op = vulten_ops::basic::op_as_str(OP);

  StatusSafePtr status(TF_NewStatus());
  auto* builder = TF_NewKernelBuilder(op.c_str(), device_type, nullptr,
                                      &BasicOps_Compute<T, OP>, nullptr);
  if (T != TF_BOOL)
    TF_KernelBuilder_TypeConstraint(builder, "T", T, status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout << " Error while registering Basic kernel with attribute T";
  TF_RegisterKernelBuilder(op.c_str(), builder, status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout << " Error while registering Basic kernel";

  // There is no apparent difference between Add and AddV2 so use the same for
  // both
  if (OP == OP_ADD) {
    StatusSafePtr status(TF_NewStatus());
    auto* builder = TF_NewKernelBuilder("AddV2", device_type, nullptr,
                                        &BasicOps_Compute<T, OP>, nullptr);
    TF_KernelBuilder_TypeConstraint(builder, "T", T, status.get());
    if (TF_OK != TF_GetCode(status.get()))
      std::cout << " Error while registering Basic kernel with attribute T";
    TF_RegisterKernelBuilder("AddV2", builder, status.get());
    if (TF_OK != TF_GetCode(status.get()))
      std::cout << " Error while registering Basic kernel";
  }
}

void RegisterDeviceBasicOps(const char* device_type) {
#define REGISTER_KERNEL(T)                               \
  RegisterBasicOpKernels<T, OP_MUL>(device_type);        \
  RegisterBasicOpKernels<T, OP_ADD>(device_type);        \
  RegisterBasicOpKernels<T, OP_SUB>(device_type);        \
  RegisterBasicOpKernels<T, OP_DIV>(device_type);        \
  RegisterBasicOpKernels<T, OP_DIV_NO_NAN>(device_type); \
  RegisterBasicOpKernels<T, OP_MAXIMUM>(device_type);    \
  RegisterBasicOpKernels<T, OP_MINIMUM>(device_type);    \
  RegisterBasicOpKernels<T, OP_DIV_REAL>(device_type);   \
  RegisterBasicOpKernels<T, OP_LESS>(device_type);       \
  RegisterBasicOpKernels<T, OP_LESS_EQUAL>(device_type); \
  RegisterBasicOpKernels<T, OP_GREATER>(device_type);    \
  RegisterBasicOpKernels<T, OP_GREATER_EQUAL>(device_type);

  CALL_ALL_BASIC_TYPES(REGISTER_KERNEL)

#define REGISTER_COMPLEX_KERNEL(T)                \
  RegisterBasicOpKernels<T, OP_ADD>(device_type); \
  RegisterBasicOpKernels<T, OP_SUB>(device_type); \
  RegisterBasicOpKernels<T, OP_MUL>(device_type); \
  RegisterBasicOpKernels<T, OP_DIV>(device_type);

  CALL_COMPLEX(REGISTER_COMPLEX_KERNEL)

#define REGISTER_BOOL_KERNEL(T)                           \
  RegisterBasicOpKernels<T, OP_LOGICAL_AND>(device_type); \
  RegisterBasicOpKernels<T, OP_LOGICAL_OR>(device_type);

  CALL_BOOL(REGISTER_BOOL_KERNEL)
}