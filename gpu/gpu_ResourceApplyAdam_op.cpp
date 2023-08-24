#include <memory>

#include "Vulten_backend/ops/resource_apply_adam/Resource_apply_adam_op.h"
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

template <TF_DataType T>
struct ResourceApplyAdamOp {
  ResourceApplyAdamOp() : locking_(false), nesterov_(false) {}
  bool locking_, nesterov_;
};

template <TF_DataType T>
void* ResourceApplyAdamOp_Create(TF_OpKernelConstruction* ctx) {
  auto kernel = new ResourceApplyAdamOp<T>();

  StatusSafePtr status(TF_NewStatus());

  unsigned char options[] = {0, 0};
  TF_OpKernelConstruction_GetAttrBool(ctx, "use_locking", &options[0],
                                      status.get());
  TF_OpKernelConstruction_GetAttrBool(ctx, "use_nesterov", &options[1],
                                      status.get());

  kernel->locking_ = options[0];
  kernel->nesterov_ = options[1];

  return kernel;
}

template <TF_DataType T>
void ResourceApplyAdamOp_Delete(void* kernel) {
  if (kernel != nullptr) {
    delete static_cast<ResourceApplyAdamOp<T>*>(kernel);
  }
}

template <TF_DataType T>
void ResourceApplyAdamOp_Compte(void* kernel, TF_OpKernelContext* ctx) {
  SCOPE_TIMER("ResourceApplyAdamOp")

  ResourceApplyAdamOp<T>* resourceApplyAdamOp =
      static_cast<ResourceApplyAdamOp<T>*>(kernel);
  StatusSafePtr status(TF_NewStatus());

  SP_Stream stream = TF_GetStream(ctx, status.get());
  vulten_backend::Instance* inst = stream->instance;

  const std::vector<int> input_to_lock = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  std::unique_ptr<TF_VariableInputLockHolder*> var_lock =
      std::unique_ptr<TF_VariableInputLockHolder*>(
          new TF_VariableInputLockHolder*);
  TF_MaybeLockVariableInputMutexesInOrder(
      ctx, resourceApplyAdamOp->locking_, false, input_to_lock.data(),
      input_to_lock.size(), &tensor_utills::copyFunc, var_lock.get(),
      status.get());

  tensor_utills::Input_tensor var = tensor_utills::get_input_tensor_from_var(
      "ResourceApplyAdamOp:var", 0, resourceApplyAdamOp->locking_, ctx,
      status.get());

  tensor_utills::Input_tensor m = tensor_utills::get_input_tensor_from_var(
      "ResourceApplyAdamOp:m", 1, resourceApplyAdamOp->locking_, ctx,
      status.get());

  tensor_utills::Input_tensor v = tensor_utills::get_input_tensor_from_var(
      "ResourceApplyAdamOp:v", 2, resourceApplyAdamOp->locking_, ctx,
      status.get());

  tensor_utills::Input_tensor beta1_power = tensor_utills::get_input_tensor(
      "ResourceApplyAdamOp:beta1_power", 3, ctx, status.get());

  tensor_utills::Input_tensor beta2_power = tensor_utills::get_input_tensor(
      "ResourceApplyAdamOp:beta2_power", 4, ctx, status.get());

  tensor_utills::Input_tensor lr = tensor_utills::get_input_tensor(
      "ResourceApplyAdamOp:lr", 5, ctx, status.get());

  tensor_utills::Input_tensor beta1 = tensor_utills::get_input_tensor(
      "ResourceApplyAdamOp:beta1", 6, ctx, status.get());

  tensor_utills::Input_tensor beta2 = tensor_utills::get_input_tensor(
      "ResourceApplyAdamOp:beta2", 7, ctx, status.get());

  tensor_utills::Input_tensor epsilon = tensor_utills::get_input_tensor(
      "ResourceApplyAdamOp:epsilon", 8, ctx, status.get());

  tensor_utills::Input_tensor grad = tensor_utills::get_input_tensor(
      "ResourceApplyAdamOp:grad", 9, ctx, status.get());

  vulten_ops::resource_apply_adam::run_op(
      inst, (vulten_ops::Data_type)T, var.vulten_tensor, m.vulten_tensor,
      v.vulten_tensor, beta1_power.vulten_tensor, beta2_power.vulten_tensor,
      lr.vulten_tensor, beta1.vulten_tensor, beta2.vulten_tensor,
      epsilon.vulten_tensor, grad.vulten_tensor,
      resourceApplyAdamOp->nesterov_);

  TF_ReleaseVariableInputLockHolder(*var_lock.get());
}

template <TF_DataType T>
void RegisterResourceApplyAdamOp(const char* device_type) {
  StatusSafePtr status(TF_NewStatus());
  auto* builder = TF_NewKernelBuilder(
      "ResourceApplyAdam", device_type, ResourceApplyAdamOp_Create<T>,
      &ResourceApplyAdamOp_Compte<T>, &ResourceApplyAdamOp_Delete<T>);
  TF_KernelBuilder_TypeConstraint(builder, "T", T, status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout
        << " Error while registering ResourceApplyAdam kernel with attribute T";
  TF_RegisterKernelBuilder("ResourceApplyAdam", builder, status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout << " Error while registering ResourceApplyAdam kernel";
}

void RegisterResourceApplyAdam(const char* device_type) {
#define REGISTER_KERNEL(T) RegisterResourceApplyAdamOp<T>(device_type);

  // CALL_ALL_BASIC_TYPES(REGISTER_KERNEL)
  REGISTER_KERNEL(TF_FLOAT)
  REGISTER_KERNEL(TF_INT32)
  REGISTER_KERNEL(TF_UINT32)
}