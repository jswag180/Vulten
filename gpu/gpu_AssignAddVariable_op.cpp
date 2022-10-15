#include <math.h>
#include <string.h>

#include <iostream>
#include <memory>
#include <vector>

#include "gpuBackend.h"
#include "gpu_variable_helpers.h"
#include "shaders/headers/shaderAddInPlace.hpp"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/kernels_experimental.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_tensor.h"
#include "vulten_device.h"

// struct StatusDeleter {
//     void operator()(TF_Status *s) {
//         if (s != nullptr) {
//             TF_DeleteStatus(s);
//         }
//     }
// };

// struct TensorDeleter {
//     void operator()(TF_Tensor *t) {
//         if (t != nullptr) {
//             TF_DeleteTensor(t);
//         }
//     }
// };

// using StatusSafePtr = std::unique_ptr<TF_Status, StatusDeleter>;
// using TensorSafePtr = std::unique_ptr<TF_Tensor, TensorDeleter>;

namespace vulten_plugin {

static std::vector<uint32_t> spirv;

template <typename T>
void AssignAddVariableOp_Compte(void *kernel, TF_OpKernelContext *ctx) {
  // utills::ScopeTimer timer("AssignAddVariableOp");
  StatusSafePtr status(TF_NewStatus());

  // TF_GetInputTensorFromVariable()
  // TF_AssignVariable()
  TF_AssignUpdateVariable(
      ctx, 0, 1, 0, 0, &varHelpers::copyFunc,
      [](TF_OpKernelContext *ctx, TF_Tensor *tensor, TF_Tensor *value, int Op) {
        StatusSafePtr status(TF_NewStatus());
        SP_Stream stream = TF_GetStream(ctx, status.get());

        // std::lock_guard<std::mutex> guard(stream->instance->testMutex);
        MutexScopeLock guard =
            MutexScopeLock(&stream->instance->mainQueueMutex);

        TensorSafePtr tensor_safe_ptr(tensor);
        auto tensor_ptr = static_cast<std::shared_ptr<kp::TensorT<float>> *>(
            TF_TensorData(tensor_safe_ptr.get()));
        // std::cout << "tensor: " << tensor_ptr << "\n";

        TensorSafePtr value_safe_ptr(value);
        auto value_ptr = static_cast<std::shared_ptr<kp::TensorT<float>> *>(
            TF_TensorData(value_safe_ptr.get()));
        // std::cout << "value: " << value_ptr << "\n";

        // std::cout <<
        // "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n";

        std::shared_ptr<kp::Algorithm> algo = stream->instance->mngr->algorithm(
            {*tensor_ptr, *value_ptr}, spirv,
            kp::Workgroup({tensor_ptr->get()->size()}));
        stream->instance->mngr->sequence(stream->instance->mainQueue)
            ->record<kp::OpAlgoDispatch>(algo)
            ->eval();

        // std::cout <<
        // "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB\n";
      },
      status.get());
  // TF_AssignRefVariable()
}

template <typename T>
void RegisterAssignAddVariableOp(const char *device_type) {
  StatusSafePtr status(TF_NewStatus());
  auto *builder =
      TF_NewKernelBuilder("AssignAddVariableOp", device_type, nullptr,
                          &AssignAddVariableOp_Compte<T>, nullptr);
  // TF_KernelBuilder_TypeConstraint(builder, "T", TF_FLOAT, status.get());
  // if (TF_OK != TF_GetCode(status.get()))
  //     std::cout << " Error while registering AssignAddVariable kernel with
  //     attribute T";
  TF_RegisterKernelBuilder("AssignAddVariableOp", builder, status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout << " Error while registering AssignAddVariable kernel";
}

}  // namespace vulten_plugin

void RegisterAssignAddVariable(const char *device_type) {
  vulten_plugin::spirv.resize(
      kp::shader_data::___shaders_AddInPlace_comp_spv_len / 4);
  memcpy(&vulten_plugin::spirv[0],
         kp::shader_data::___shaders_AddInPlace_comp_spv,
         kp::shader_data::___shaders_AddInPlace_comp_spv_len);

  vulten_plugin::RegisterAssignAddVariableOp<float>(device_type);
}
