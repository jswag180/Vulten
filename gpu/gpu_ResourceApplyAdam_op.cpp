#include <math.h>
#include <string.h>

#include <iostream>
#include <memory>
#include <vector>

#include "gpuBackend.h"
#include "gpu_variable_helpers.h"
#include "shaders/headers/shaderApplyAdam.hpp"
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

//#define debugAdam 1

namespace vulten_plugin {

static std::vector<uint32_t> spirv;

template <class T>
struct ResourceApplyAdamOp {
  ResourceApplyAdamOp() : locking_(false), nesterov_(false) {}
  bool locking_, nesterov_;
};

template <typename T>
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

template <typename T>
void ResourceApplyAdamOp_Delete(void* kernel) {
  if (kernel != nullptr) {
    delete static_cast<ResourceApplyAdamOp<T>*>(kernel);
  }
}

template <typename T>
void ResourceApplyAdamOp_Compte(void* kernel, TF_OpKernelContext* ctx) {
  ResourceApplyAdamOp<T>* resourceApplyAdamOp =
      static_cast<ResourceApplyAdamOp<T>*>(kernel);
  StatusSafePtr status(TF_NewStatus());

  TF_Tensor** var_ref = new TF_Tensor*;
  TF_GetInputTensorFromVariable(ctx, 0, resourceApplyAdamOp->locking_, 0, 0,
                                &varHelpers::copyFunc, var_ref, status.get());
  auto var_ptr = static_cast<std::shared_ptr<kp::TensorT<float>>*>(
      TF_TensorData(*var_ref));

  TF_Tensor** m_ref = new TF_Tensor*;
  TF_GetInputTensorFromVariable(ctx, 1, resourceApplyAdamOp->locking_, 0, 0,
                                &varHelpers::copyFunc, m_ref, status.get());
  auto m_ptr =
      static_cast<std::shared_ptr<kp::TensorT<float>>*>(TF_TensorData(*m_ref));

  TF_Tensor** v_ref = new TF_Tensor*;
  TF_GetInputTensorFromVariable(ctx, 2, resourceApplyAdamOp->locking_, 0, 0,
                                &varHelpers::copyFunc, v_ref, status.get());
  auto v_ptr =
      static_cast<std::shared_ptr<kp::TensorT<float>>*>(TF_TensorData(*v_ref));

  TF_Tensor* beta1_power = nullptr;
  TF_GetInput(ctx, 3, &beta1_power, status.get());
  TensorSafePtr beta1_power_safe_ptr(beta1_power);
  if (TF_GetCode(status.get()) != TF_OK) {
    TF_OpKernelContext_Failure(ctx, status.get());
    return;
  }
  auto beta1_power_ptr = static_cast<std::shared_ptr<kp::TensorT<float>>*>(
      TF_TensorData(beta1_power_safe_ptr.get()));

  TF_Tensor* beta2_power = nullptr;
  TF_GetInput(ctx, 4, &beta2_power, status.get());
  TensorSafePtr beta2_power_safe_ptr(beta2_power);
  if (TF_GetCode(status.get()) != TF_OK) {
    TF_OpKernelContext_Failure(ctx, status.get());
    return;
  }
  auto beta2_power_ptr = static_cast<std::shared_ptr<kp::TensorT<float>>*>(
      TF_TensorData(beta2_power_safe_ptr.get()));

  TF_Tensor* lr = nullptr;
  TF_GetInput(ctx, 5, &lr, status.get());
  TensorSafePtr lr_safe_ptr(lr);
  if (TF_GetCode(status.get()) != TF_OK) {
    TF_OpKernelContext_Failure(ctx, status.get());
    return;
  }
  auto lr_ptr = static_cast<std::shared_ptr<kp::TensorT<float>>*>(
      TF_TensorData(lr_safe_ptr.get()));

  TF_Tensor* beta1 = nullptr;
  TF_GetInput(ctx, 6, &beta1, status.get());
  TensorSafePtr beta1_safe_ptr(beta1);
  if (TF_GetCode(status.get()) != TF_OK) {
    TF_OpKernelContext_Failure(ctx, status.get());
    return;
  }
  auto beta1_ptr = static_cast<std::shared_ptr<kp::TensorT<float>>*>(
      TF_TensorData(beta1_safe_ptr.get()));

  TF_Tensor* beta2 = nullptr;
  TF_GetInput(ctx, 7, &beta2, status.get());
  TensorSafePtr beta2_safe_ptr(beta2);
  if (TF_GetCode(status.get()) != TF_OK) {
    TF_OpKernelContext_Failure(ctx, status.get());
    return;
  }
  auto beta2_ptr = static_cast<std::shared_ptr<kp::TensorT<float>>*>(
      TF_TensorData(beta2_safe_ptr.get()));

  TF_Tensor* epsilon = nullptr;
  TF_GetInput(ctx, 8, &epsilon, status.get());
  TensorSafePtr epsilon_safe_ptr(epsilon);
  if (TF_GetCode(status.get()) != TF_OK) {
    TF_OpKernelContext_Failure(ctx, status.get());
    return;
  }
  auto epsilon_ptr = static_cast<std::shared_ptr<kp::TensorT<float>>*>(
      TF_TensorData(epsilon_safe_ptr.get()));

  TF_Tensor* grad = nullptr;
  TF_GetInput(ctx, 9, &grad, status.get());
  TensorSafePtr grad_safe_ptr(grad);
  if (TF_GetCode(status.get()) != TF_OK) {
    TF_OpKernelContext_Failure(ctx, status.get());
    return;
  }
  auto grad_ptr = static_cast<std::shared_ptr<kp::TensorT<float>>*>(
      TF_TensorData(grad_safe_ptr.get()));

#ifdef debugAdam
  std::cout << "var: \n";
  for (int i = 0; i < var_ptr->get()->size(); i++) {
    std::cout << var_ptr->get()->data()[i] << "\n";
  }

  std::cout << "m: \n";
  for (int i = 0; i < m_ptr->get()->size(); i++) {
    std::cout << m_ptr->get()->data()[i] << "\n";
  }

  std::cout << "v: \n";
  for (int i = 0; i < v_ptr->get()->size(); i++) {
    std::cout << v_ptr->get()->data()[i] << "\n";
  }

  std::cout << "beta1_power: \n";
  std::cout << beta1_power_ptr->get()->data()[0] << "\n";

  std::cout << "beta2_power: \n";
  std::cout << beta2_power_ptr->get()->data()[0] << "\n";

  std::cout << "lr: \n";
  std::cout << lr_ptr->get()->data()[0] << "\n";

  std::cout << "beta1: \n";
  std::cout << beta1_ptr->get()->data()[0] << "\n";

  std::cout << "beta2: \n";
  std::cout << beta2_ptr->get()->data()[0] << "\n";

  std::cout << "epsilon: \n";
  std::cout << epsilon_ptr->get()->data()[0] << "\n";

  std::cout << "grad: \n";
  for (int i = 0; i < grad_ptr->get()->size(); i++) {
    std::cout << grad_ptr->get()->data()[i] << "\n";
  }

  std::cout << "use_nesterov: " << resourceApplyAdamOp->nesterov_ << "\n";
#endif

  SP_Stream stream = TF_GetStream(ctx, status.get());
  // std::lock_guard<std::mutex> guard(stream->instance->testMutex);
  MutexScopeLock guard = MutexScopeLock(&stream->instance->mainQueueMutex);

  std::shared_ptr<kp::Algorithm> algo = stream->instance->mngr->algorithm(
      {*var_ptr, *m_ptr, *v_ptr, *beta1_power_ptr, *beta2_power_ptr, *lr_ptr,
       *beta1_ptr, *beta2_ptr, *epsilon_ptr, *grad_ptr},
      spirv, kp::Workgroup({var_ptr->get()->size()}),
      std::vector<uint>{resourceApplyAdamOp->nesterov_}, {});

  stream->instance->mngr->sequence(stream->instance->mainQueue)
      ->record<kp::OpAlgoDispatch>(algo)
      ->eval();

  // TF_AssignVariable()
  // TF_AssignUpdateVariable()
  delete var_ref;
  delete m_ref;
  delete v_ref;
}

template <typename T>
void RegisterResourceApplyAdamOp(const char* device_type) {
  StatusSafePtr status(TF_NewStatus());
  auto* builder = TF_NewKernelBuilder(
      "ResourceApplyAdam", device_type, ResourceApplyAdamOp_Create<T>,
      &ResourceApplyAdamOp_Compte<T>, &ResourceApplyAdamOp_Delete<T>);
  TF_KernelBuilder_TypeConstraint(builder, "T", TF_FLOAT, status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout
        << " Error while registering ResourceApplyAdam kernel with attribute T";
  TF_RegisterKernelBuilder("ResourceApplyAdam", builder, status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout << " Error while registering ResourceApplyAdam kernel";
}

}  // namespace vulten_plugin

void RegisterResourceApplyAdam(const char* device_type) {
  vulten_plugin::spirv.resize(
      kp::shader_data::___shaders_ApplyAdam_comp_spv_len / 4);
  memcpy(&vulten_plugin::spirv[0],
         kp::shader_data::___shaders_ApplyAdam_comp_spv,
         kp::shader_data::___shaders_ApplyAdam_comp_spv_len);

  vulten_plugin::RegisterResourceApplyAdamOp<float>(device_type);
}
