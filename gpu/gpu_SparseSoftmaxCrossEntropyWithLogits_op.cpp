#include <chrono>
#include <iostream>
#include <memory>
#include <vector>

#include "Vulten_backend/Vulten_backend.h"
#include "Vulten_backend/ops/Vulten_backend_ops.h"
#include "Vulten_backend/ops/basic/Basic_ops.h"
#include "Vulten_backend/ops/multiFunc/MultiFunc_op.h"
#include "Vulten_backend/ops/reduce/Reduce_op.h"
#include "Vulten_backend/ops/xent/Xent_op.h"
#include "absl/container/inlined_vector.h"
#include "scope_timer.h"
#include "tensor_utills.h"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_tensor.h"
#include "vulten_device.h"

template <TF_DataType T>
void SparseSoftmaxCrossEntropyWithLogitsOp_Compute(void* kernel,
                                                   TF_OpKernelContext* ctx) {
  SCOPE_TIMER("SparseSoftmaxCrossEntropyWithLogitsOp")

  StatusSafePtr status(TF_NewStatus());

  tensor_utills::Input_tensor features = tensor_utills::get_input_tensor(
      "SparseSoftmaxCrossEntropyWithLogitsOp:features", 0, ctx, status.get());

  tensor_utills::Input_tensor labels = tensor_utills::get_input_tensor(
      "SparseSoftmaxCrossEntropyWithLogitsOp:labels", 1, ctx, status.get());

  absl::InlinedVector<int64_t, 4> loss_dims =
      absl::InlinedVector<int64_t, 4>(1);
  loss_dims[0] = features.dims[0];
  tensor_utills::Output_tensor loss = tensor_utills::make_output_tensor(
      "SparseSoftmaxCrossEntropyWithLogitsOp:loss", 0, loss_dims, ctx,
      status.get());

  tensor_utills::Output_tensor backprop = tensor_utills::make_output_tensor(
      "SparseSoftmaxCrossEntropyWithLogitsOp:backprop", 1, features.dims, ctx,
      status.get());

  if (features.is_empty) {
    return;
  }

  SP_Stream stream = TF_GetStream(ctx, status.get());
  vulten_backend::Instance* inst = stream->instance;

  vulten_ops::Reduce_op* reduce_op = nullptr;
  std::string op_cache_reduce = "Reduce";
  vulten_ops::Basic_op* basic_op = nullptr;
  std::string op_cache_basic = "Basic";
  vulten_ops::MultiFunc_op* multiFunc_op = nullptr;
  std::string op_cache_multiFunc = "MultiFunc";
  vulten_ops::Xent_op* xent_op = nullptr;
  std::string op_cache_xent = "Xent";
  inst->main_queue_mutex.lock();
  if (inst->op_chache.find(op_cache_reduce) == inst->op_chache.end()) {
    inst->op_chache[op_cache_reduce] =
        (vulten_ops::Vulten_op*)new vulten_ops::Reduce_op(inst);
  }
  reduce_op = (vulten_ops::Reduce_op*)inst->op_chache[op_cache_reduce];
  if (inst->op_chache.find(op_cache_basic) == inst->op_chache.end()) {
    inst->op_chache[op_cache_basic] =
        (vulten_ops::Vulten_op*)new vulten_ops::Basic_op(inst);
  }
  basic_op = (vulten_ops::Basic_op*)inst->op_chache[op_cache_basic];
  if (inst->op_chache.find(op_cache_multiFunc) == inst->op_chache.end()) {
    inst->op_chache[op_cache_multiFunc] =
        (vulten_ops::Vulten_op*)new vulten_ops::MultiFunc_op(inst);
  }
  multiFunc_op = (vulten_ops::MultiFunc_op*)inst->op_chache[op_cache_multiFunc];
  if (inst->op_chache.find(op_cache_xent) == inst->op_chache.end()) {
    inst->op_chache[op_cache_xent] =
        (vulten_ops::Vulten_op*)new vulten_ops::Xent_op(inst);
  }
  xent_op = (vulten_ops::Xent_op*)inst->op_chache[op_cache_xent];
  inst->main_queue_mutex.unlock();

  // maxFeatures
  auto max_features = std::unique_ptr<vulten_backend::Device_buffer>(
      inst->create_device_buffer(TF_DataTypeSize(T) * features.dims[0]));
  int64_t max_features_dims[2] = {features.dims[0], 1};
  vulten_ops::Vulten_tensor max_features_tensor =
      vulten_ops::Vulten_tensor(max_features.get(), 1, max_features_dims);

  std::vector<int32_t> axis_vec = {1};
  reduce_op->run_op((vulten_ops::Data_type)T, features.vulten_tensor, axis_vec,
                    max_features_tensor, OP_MAX);

  // features - maxFeatures
  max_features_tensor.num_dims = 2;
  basic_op->run_op((vulten_ops::Data_type)T, OP_SUB, features.vulten_tensor,
                   max_features_tensor, backprop.vulten_tensor);

  // exp(backprop).sum(1)
  // exp
  auto scratch_exp = std::unique_ptr<vulten_backend::Device_buffer>(
      inst->create_device_buffer(backprop.vulten_tensor.buffer->buffer_size));
  vulten_ops::Vulten_tensor scratch_exp_tensor = vulten_ops::Vulten_tensor(
      scratch_exp.get(), backprop.vulten_tensor.num_dims,
      backprop.vulten_tensor.dims);

  multiFunc_op->run_op((vulten_ops::Data_type)T, backprop.vulten_tensor,
                       scratch_exp_tensor, OP_EXP);

  // sum(1)
  auto scratch = std::unique_ptr<vulten_backend::Device_buffer>(
      inst->create_device_buffer(TF_DataTypeSize(T) * features.dims[0]));
  int64_t scratch_dims[1] = {features.dims[0]};
  vulten_ops::Vulten_tensor scratch_tensor =
      vulten_ops::Vulten_tensor(scratch.get(), 1, scratch_dims);

  reduce_op->run_op((vulten_ops::Data_type)T, scratch_exp_tensor, axis_vec,
                    scratch_tensor, OP_SUM);

  // XentUills(OP_LOSS)
  auto loss_fat = std::unique_ptr<vulten_backend::Device_buffer>(
      inst->create_device_buffer(backprop.vulten_tensor.buffer->buffer_size));
  vulten_ops::Vulten_tensor loss_fat_tensor =
      vulten_ops::Vulten_tensor(loss_fat.get(), backprop.vulten_tensor.num_dims,
                                backprop.vulten_tensor.dims);

  xent_op->run_op((vulten_ops::Data_type)T,
                  (vulten_ops::Data_type)TF_TensorType(labels.tf_tensor),
                  scratch_tensor, backprop.vulten_tensor, labels.vulten_tensor,
                  loss_fat_tensor, OP_LOSS);
  reduce_op->run_op((vulten_ops::Data_type)T, loss_fat_tensor, axis_vec,
                    loss.vulten_tensor, OP_SUM);

  xent_op->run_op((vulten_ops::Data_type)T,
                  (vulten_ops::Data_type)TF_TensorType(labels.tf_tensor),
                  scratch_tensor, backprop.vulten_tensor, labels.vulten_tensor,
                  backprop.vulten_tensor, OP_GRAD);
}

template <TF_DataType T>
void RegisterSparseSoftmaxCrossEntropyWithLogitsOpKernel(
    const char* device_type) {
  StatusSafePtr status(TF_NewStatus());
  auto* builder = TF_NewKernelBuilder(
      "SparseSoftmaxCrossEntropyWithLogits", device_type, nullptr,
      &SparseSoftmaxCrossEntropyWithLogitsOp_Compute<T>, nullptr);
  TF_KernelBuilder_TypeConstraint(builder, "T", T, status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout << " Error while registering SparseSoftmaxCrossEntropyWithLogits "
                 "kernel with attribute T";
  TF_RegisterKernelBuilder("SparseSoftmaxCrossEntropyWithLogits", builder,
                           status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout << " Error while registering SparseSoftmaxCrossEntropyWithLogits "
                 "kernel";
}

void RegisterDeviceSparseSoftmaxCrossEntropyWithLogits(
    const char* device_type) {
#define REGISTER_KERNEL(T) \
  RegisterSparseSoftmaxCrossEntropyWithLogitsOpKernel<T>(device_type);

  REGISTER_KERNEL(TF_FLOAT)
}