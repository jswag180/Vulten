#include "Vulten_backend/ops/assign_add_sub/Assign_add_sub_op.h"
#include "absl/container/inlined_vector.h"
#include "scope_timer.h"
#include "tensor_utills.h"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/kernels_experimental.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_tensor.h"
#include "vulten_device.h"

void addSubUpdate(TF_OpKernelContext *ctx, TF_Tensor *tensor, TF_Tensor *value,
                  int Op) {
  StatusSafePtr status(TF_NewStatus());
  SP_Stream stream = TF_GetStream(ctx, status.get());
  vulten_backend::Instance *inst = stream->instance;

  TensorSafePtr tensor_safe_ptr(tensor);
  auto tensor_ptr = VOID_TO_DEVICE_BUFFER(TF_TensorData(tensor_safe_ptr.get()));
  absl::InlinedVector<int64_t, 4> tensor_dims(
      TF_NumDims(tensor_safe_ptr.get()));
  for (auto i = 0; i < TF_NumDims(tensor_safe_ptr.get()); ++i) {
    tensor_dims[i] = TF_Dim(tensor_safe_ptr.get(), i);
  }
  vulten_ops::Vulten_tensor input_tensor = vulten_ops::Vulten_tensor(
      tensor_ptr, tensor_dims.size(), tensor_dims.data());

  TensorSafePtr value_safe_ptr(value);
  auto value_ptr = VOID_TO_DEVICE_BUFFER(TF_TensorData(value_safe_ptr.get()));
  vulten_ops::Vulten_tensor value_tensor = vulten_ops::Vulten_tensor(
      value_ptr, tensor_dims.size(), tensor_dims.data());

  vulten_ops::assign_add_sub::run_op(
      inst, (vulten_ops::Data_type)TF_TensorType(tensor_safe_ptr.get()),
      input_tensor, value_tensor, Op);
}

template <TF_DataType T, int32_t OP>
void AssignAddSubVariableOp_Compte(void *kernel, TF_OpKernelContext *ctx) {
  if (OP == ADD) {
    SCOPE_TIMER("AssignAddVariableOp")
  } else if (OP == SUB) {
    SCOPE_TIMER("AssignSubVariableOp")
  }

  StatusSafePtr status(TF_NewStatus());

  TF_AssignUpdateVariable(ctx, 0, 1, OP, 0, &tensor_utills::copyFunc,
                          &addSubUpdate, status.get());
}

template <TF_DataType T, int32_t OP>
void RegisterAssignAddSubVariableOp(const char *device_type) {
  StatusSafePtr status(TF_NewStatus());

  std::string op_str = "";
  if (OP == ADD) {
    op_str = "AssignAddVariableOp";
  } else if (OP == SUB) {
    op_str = "AssignSubVariableOp";
  }

  auto *builder =
      TF_NewKernelBuilder(op_str.c_str(), device_type, nullptr,
                          &AssignAddSubVariableOp_Compte<T, OP>, nullptr);
  TF_KernelBuilder_TypeConstraint(builder, "dtype", T, status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout << " Error while registering AssignAddSubVariable kernel with "
                 "attribute T";
  TF_RegisterKernelBuilder(op_str.c_str(), builder, status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout << " Error while registering AssignAddSubVariable kernel";
}

void RegisterAssignAddSubVariable(const char *device_type) {
#define REGISTER_KERNEL_ADD(T) \
  RegisterAssignAddSubVariableOp<T, ADD>(device_type);
#define REGISTER_KERNEL_SUB(T) \
  RegisterAssignAddSubVariableOp<T, SUB>(device_type);

  CALL_ALL_TYPES(REGISTER_KERNEL_ADD)
  CALL_ALL_TYPES(REGISTER_KERNEL_SUB)
}
