#include "Vulten_backend/Vulten_utills.h"
#include "Vulten_backend/ops/cast/Cast_op.h"
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

  vulten_ops::cast::run_op(inst, (vulten_ops::Data_type)S,
                           (vulten_ops::Data_type)D, input.vulten_tensor,
                           output.vulten_tensor);
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
#define REGISTER_TYPE_8BIT(s)                    \
  RegisterCastOpKernel<s, TF_BOOL>(device_type); \
  RegisterCastOpKernel<s, TF_INT8>(device_type); \
  RegisterCastOpKernel<s, TF_UINT8>(device_type);
#define REGISTER_TYPE_INT16(s)                    \
  RegisterCastOpKernel<s, TF_INT16>(device_type); \
  RegisterCastOpKernel<s, TF_UINT16>(device_type);
#define REGISTER_TYPE_INT64(s)                    \
  RegisterCastOpKernel<s, TF_INT64>(device_type); \
  RegisterCastOpKernel<s, TF_UINT64>(device_type);
#define REGISTER_TYPE_FLOAT16(s) RegisterCastOpKernel<s, TF_HALF>(device_type);
#define REGISTER_TYPE_FLOAT64(s)                   \
  RegisterCastOpKernel<s, TF_DOUBLE>(device_type); \
  RegisterCastOpKernel<s, TF_COMPLEX128>(device_type);
#define REGISTER_TYPE(s)                                    \
  RegisterCastOpKernel<s, TF_FLOAT>(device_type);           \
  RegisterCastOpKernel<s, TF_INT32>(device_type);           \
  RegisterCastOpKernel<s, TF_UINT32>(device_type);          \
  RegisterCastOpKernel<s, TF_COMPLEX64>(device_type);       \
  if (!vulten_utills::get_env_bool(VULTEN_DISABLE_INT64))   \
    REGISTER_TYPE_INT64(s)                                  \
  if (!vulten_utills::get_env_bool(VULTEN_DISABLE_FLOAT64)) \
    REGISTER_TYPE_FLOAT64(s)                                \
  if (!vulten_utills::get_env_bool(VULTEN_DISABLE_FLOAT16)) \
    REGISTER_TYPE_FLOAT16(s)                                \
  if (!vulten_utills::get_env_bool(VULTEN_DISABLE_INT16))   \
    REGISTER_TYPE_INT16(s)                                  \
  if (!vulten_utills::get_env_bool(VULTEN_DISABLE_INT8)) REGISTER_TYPE_8BIT(s)

  CALL_ALL_TYPES(REGISTER_TYPE)

  CALL_BOOL(REGISTER_TYPE)
}