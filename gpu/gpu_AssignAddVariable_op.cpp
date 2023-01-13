#include "Vulten_backend/ops/Assign_add_op.h"
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
void AssignAddVariableOp_Compte(void *kernel, TF_OpKernelContext *ctx) {
  SCOPE_TIMER("AssignAddVariableOp")

  StatusSafePtr status(TF_NewStatus());

  TF_AssignUpdateVariable(
      ctx, 0, 1, 0, 0, &tensor_utills::copyFunc,
      [](TF_OpKernelContext *ctx, TF_Tensor *tensor, TF_Tensor *value, int Op) {
        StatusSafePtr status(TF_NewStatus());
        SP_Stream stream = TF_GetStream(ctx, status.get());
        vulten_backend::Instance *inst = stream->instance;

        vulten_ops::Assign_add_op<(vulten_ops::Data_type)T> *assign_add =
            nullptr;
        std::string op_cache_name = "Assign_add_" + std::to_string(T);
        inst->main_queue_mutex.lock();
        if (inst->op_chache.find(op_cache_name) == inst->op_chache.end()) {
          inst->op_chache[op_cache_name] =
              (vulten_ops::Vulten_op *)new vulten_ops::Assign_add_op<(
                  vulten_ops::Data_type)T>(inst);
        }
        assign_add = (vulten_ops::Assign_add_op<(vulten_ops::Data_type)T> *)
                         inst->op_chache[op_cache_name];
        inst->main_queue_mutex.unlock();

        TensorSafePtr tensor_safe_ptr(tensor);
        auto tensor_ptr =
            VOID_TO_DEVICE_BUFFER(TF_TensorData(tensor_safe_ptr.get()));
        absl::InlinedVector<int64_t, 4> tensor_dims(
            TF_NumDims(tensor_safe_ptr.get()));
        for (auto i = 0; i < TF_NumDims(tensor_safe_ptr.get()); ++i) {
          tensor_dims[i] = TF_Dim(tensor_safe_ptr.get(), i);
        }
        vulten_ops::Vulten_tensor input_tensor = vulten_ops::Vulten_tensor(
            tensor_ptr, tensor_dims.size(), tensor_dims.data());

        TensorSafePtr value_safe_ptr(value);
        auto value_ptr =
            VOID_TO_DEVICE_BUFFER(TF_TensorData(value_safe_ptr.get()));
        vulten_ops::Vulten_tensor value_tensor = vulten_ops::Vulten_tensor(
            value_ptr, tensor_dims.size(), tensor_dims.data());

        assign_add->run_op(input_tensor, value_tensor);
      },
      status.get());
}

template <TF_DataType T>
void RegisterAssignAddVariableOp(const char *device_type) {
  StatusSafePtr status(TF_NewStatus());
  auto *builder =
      TF_NewKernelBuilder("AssignAddVariableOp", device_type, nullptr,
                          &AssignAddVariableOp_Compte<T>, nullptr);
  TF_KernelBuilder_TypeConstraint(builder, "dtype", T, status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout
        << " Error while registering AssignAddVariable kernel with attribute T";
  TF_RegisterKernelBuilder("AssignAddVariableOp", builder, status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout << " Error while registering AssignAddVariable kernel";
}

void RegisterAssignAddVariable(const char *device_type) {
#define REGISTER_KERNEL(T) RegisterAssignAddVariableOp<T>(device_type);

  CALL_ALL_BASIC_TYPES(REGISTER_KERNEL)
}
