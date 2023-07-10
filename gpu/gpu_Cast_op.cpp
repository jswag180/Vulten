#include "Vulten_backend/ops/Cast_op.h"
#include "absl/container/inlined_vector.h"
#include "scope_timer.h"
#include "tensor_utills.h"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "vulten_device.h"

template <TF_DataType S, TF_DataType D>
void CastOp_Compute(void* kernel, TF_OpKernelContext* ctx) {
  SCOPE_TIMER("CastOp " + std::to_string(static_cast<int>(S)) + " -> " +
              std::to_string(static_cast<int>(D)))

  StatusSafePtr status(TF_NewStatus());

  tensor_utills::Input_tensor input =
      tensor_utills::get_input_tensor("CastOp:input", 0, ctx, status.get());

  tensor_utills::Output_tensor output = tensor_utills::make_output_tensor(
      "CastOp:output", 0, input.dims, ctx, status.get());

  if (input.is_empty) {
    return;
  }

  SP_Stream stream = TF_GetStream(ctx, status.get());
  vulten_backend::Instance* inst = stream->instance;

  vulten_ops::Cast_op* cast_op = nullptr;
  std::string op_cache_name = "Cast";
  inst->main_queue_mutex.lock();
  if (inst->op_chache.find(op_cache_name) == inst->op_chache.end()) {
    inst->op_chache[op_cache_name] =
        (vulten_ops::Vulten_op*)new vulten_ops::Cast_op(inst);
  }
  cast_op = (vulten_ops::Cast_op*)inst->op_chache[op_cache_name];
  inst->main_queue_mutex.unlock();

  cast_op->run_op((vulten_ops::Data_type)S, (vulten_ops::Data_type)D,
                  input.vulten_tensor, output.vulten_tensor);
}

template <TF_DataType S, TF_DataType D>
void RegisterCastOpKernel(const char* device_type) {
  StatusSafePtr status(TF_NewStatus());
  auto* builder = TF_NewKernelBuilder("Cast", device_type, nullptr,
                                      &CastOp_Compute<S, D>, nullptr);

  TF_KernelBuilder_TypeConstraint(builder, "SrcT", S, status.get());
  TF_KernelBuilder_TypeConstraint(builder, "DstT", D, status.get());

  if (TF_OK != TF_GetCode(status.get()))
    std::cout << " Error while registering Cast kernel with attribute T";
  TF_RegisterKernelBuilder("Cast", builder, status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout << " Error while registering Cast kernel";
}

void RegisterDeviceCast(const char* device_type) {
#define REGISTER_TYPE(s)                               \
  RegisterCastOpKernel<s, TF_FLOAT>(device_type);      \
  RegisterCastOpKernel<s, TF_INT32>(device_type);      \
  RegisterCastOpKernel<s, TF_UINT32>(device_type);     \
  RegisterCastOpKernel<s, TF_INT64>(device_type);      \
  RegisterCastOpKernel<s, TF_UINT64>(device_type);     \
  RegisterCastOpKernel<s, TF_INT8>(device_type);       \
  RegisterCastOpKernel<s, TF_UINT8>(device_type);      \
  RegisterCastOpKernel<s, TF_DOUBLE>(device_type);     \
  RegisterCastOpKernel<s, TF_HALF>(device_type);       \
  RegisterCastOpKernel<s, TF_INT16>(device_type);      \
  RegisterCastOpKernel<s, TF_UINT16>(device_type);     \
  RegisterCastOpKernel<s, TF_COMPLEX64>(device_type);  \
  RegisterCastOpKernel<s, TF_COMPLEX128>(device_type); \
  RegisterCastOpKernel<s, TF_BOOL>(device_type);

  CALL_ALL_TYPES(REGISTER_TYPE)

  REGISTER_TYPE(TF_BOOL)
}