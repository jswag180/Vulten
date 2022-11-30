#include <chrono>
#include <iostream>
#include <map>
#include <memory>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "gpuBackend.h"
#include "shaders/headers/Cast/Cast.h"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "vulten_device.h"

struct StatusDeleter {
  void operator()(TF_Status* s) {
    if (s != nullptr) {
      TF_DeleteStatus(s);
    }
  }
};

struct TensorDeleter {
  void operator()(TF_Tensor* t) {
    if (t != nullptr) {
      TF_DeleteTensor(t);
    }
  }
};

using StatusSafePtr = std::unique_ptr<TF_Status, StatusDeleter>;
using TensorSafePtr = std::unique_ptr<TF_Tensor, TensorDeleter>;

namespace vulten_plugin {

struct CastOp {
  CastOp() : truncate_(false), dstT_(TF_FLOAT) {}
  bool truncate_;
  TF_DataType dstT_;
};

template <TF_DataType S, TF_DataType D>
void* CastOp_Create(TF_OpKernelConstruction* ctx) {
  auto kernel = new CastOp();

  StatusSafePtr status(TF_NewStatus());

  TF_OpKernelConstruction_GetAttrType(ctx, "DstT", &kernel->dstT_,
                                      status.get());

  return kernel;
}

void CastOp_Delete(void* kernel) {
  if (kernel != nullptr) {
    delete static_cast<CastOp*>(kernel);
  }
}

template <TF_DataType S, TF_DataType D, const std::vector<uint32_t>* spirv>
void CastOp_Compute(void* kernel, TF_OpKernelContext* ctx) {
  // utills::ScopeTimer timer("CastOp");

  CastOp* castOp = static_cast<CastOp*>(kernel);

  StatusSafePtr status(TF_NewStatus());

  TF_Tensor* input = nullptr;
  TF_GetInput(ctx, 0, &input, status.get());
  TensorSafePtr input_safe_ptr(input);
  if (TF_GetCode(status.get()) != TF_OK) {
    TF_OpKernelContext_Failure(ctx, status.get());
    return;
  }
  if (TF_TensorElementCount(input_safe_ptr.get()) == 0) return;
  absl::InlinedVector<int64_t, 4> dims(TF_NumDims(input_safe_ptr.get()));
  for (auto i = 0; i < TF_NumDims(input_safe_ptr.get()); ++i) {
    dims[i] = TF_Dim(input_safe_ptr.get(), i);
  }

  TensorSafePtr output_safe_ptr(TF_AllocateOutput(
      ctx, 0, TF_ExpectedOutputDataType(ctx, 0), dims.data(), dims.size(),
      TF_TensorElementCount(input_safe_ptr.get()) * TF_DataTypeSize(D),
      status.get()));
  if (TF_GetCode(status.get()) != TF_OK) {
    TF_OpKernelContext_Failure(ctx, status.get());
    return;
  }

  SP_Stream stream = TF_GetStream(ctx, status.get());
  MutexScopeLock guard = MutexScopeLock(&stream->instance->mainQueueMutex);

  auto in_ptr = static_cast<std::shared_ptr<kp::TensorT<float>>*>(
      TF_TensorData(input_safe_ptr.get()));
  auto out_ptr = static_cast<std::shared_ptr<kp::TensorT<float>>*>(
      TF_TensorData(output_safe_ptr.get()));

  std::shared_ptr<kp::Algorithm> algo = stream->instance->mngr->algorithm(
      {*in_ptr, *out_ptr}, *spirv, kp::Workgroup({in_ptr->get()->size()}), {},
      {});

  stream->instance->mngr->sequence(stream->instance->mainQueue)
      ->record<kp::OpAlgoDispatch>(algo)
      ->eval();
}

template <TF_DataType S, TF_DataType D, const std::vector<uint32_t>* spirv>
void RegisterCastOpKernel(const char* device_type) {
  StatusSafePtr status(TF_NewStatus());
  auto* builder =
      TF_NewKernelBuilder("Cast", device_type, CastOp_Create<S, D>,
                          &CastOp_Compute<S, D, spirv>, &CastOp_Delete);

  TF_KernelBuilder_TypeConstraint(builder, "SrcT", S, status.get());
  TF_KernelBuilder_TypeConstraint(builder, "DstT", D, status.get());

  if (TF_OK != TF_GetCode(status.get()))
    std::cout << " Error while registering Cast kernel with attribute T";
  TF_RegisterKernelBuilder("Cast", builder, status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout << " Error while registering Cast kernel";
}

}  // namespace vulten_plugin

#define REGISTER_TYPE_PAIR(s, d, c) \
  vulten_plugin::RegisterCastOpKernel<s, d, &c>(device_type);

void RegisterDeviceCast(const char* device_type) {
#ifdef CAST_FLOAT_FLOAT
  REGISTER_TYPE_PAIR(TF_FLOAT, TF_FLOAT, shader::Cast_float_float)
#endif
#ifdef CAST_FLOAT_INT
  REGISTER_TYPE_PAIR(TF_FLOAT, TF_INT32, shader::Cast_float_int)
#endif
#ifdef CAST_FLOAT_UINT
  REGISTER_TYPE_PAIR(TF_FLOAT, TF_UINT32, shader::Cast_float_uint)
#endif
#ifdef CAST_FLOAT_INT64_T
  REGISTER_TYPE_PAIR(TF_FLOAT, TF_INT64, shader::Cast_float_int64_t)
#endif
#ifdef CAST_FLOAT_UINT64_T
  REGISTER_TYPE_PAIR(TF_FLOAT, TF_UINT64, shader::Cast_float_uint64_t)
#endif
#ifdef CAST_FLOAT_INT8_T
  REGISTER_TYPE_PAIR(TF_FLOAT, TF_INT8, shader::Cast_float_int8_t)
#endif
#ifdef CAST_FLOAT_UINT8_T
  REGISTER_TYPE_PAIR(TF_FLOAT, TF_UINT8, shader::Cast_float_uint8_t)
#endif
#ifdef CAST_FLOAT_DOUBLE
  REGISTER_TYPE_PAIR(TF_FLOAT, TF_DOUBLE, shader::Cast_float_double)
#endif

#ifdef CAST_INT_INT
  REGISTER_TYPE_PAIR(TF_INT32, TF_INT32, shader::Cast_int_int)
#endif
#ifdef CAST_INT_FLOAT
  REGISTER_TYPE_PAIR(TF_INT32, TF_FLOAT, shader::Cast_int_float)
#endif
#ifdef CAST_INT_UINT
  REGISTER_TYPE_PAIR(TF_INT32, TF_UINT32, shader::Cast_int_uint)
#endif
#ifdef CAST_INT_INT64_T
  REGISTER_TYPE_PAIR(TF_INT32, TF_INT64, shader::Cast_int_int64_t)
#endif
#ifdef CAST_INT_UINT64_T
  REGISTER_TYPE_PAIR(TF_INT32, TF_UINT64, shader::Cast_int_uint64_t)
#endif
#ifdef CAST_INT_INT8_T
  REGISTER_TYPE_PAIR(TF_INT32, TF_INT8, shader::Cast_int_int8_t)
#endif
#ifdef CAST_INT_UINT8_T
  REGISTER_TYPE_PAIR(TF_INT32, TF_UINT8, shader::Cast_int_uint8_t)
#endif
#ifdef CAST_INT_DOUBLE
  REGISTER_TYPE_PAIR(TF_INT32, TF_DOUBLE, shader::Cast_int_double)
#endif

#ifdef CAST_UINT_UINT
  REGISTER_TYPE_PAIR(TF_UINT32, TF_UINT32, shader::Cast_uint_uint)
#endif
#ifdef CAST_UINT_FLOAT
  REGISTER_TYPE_PAIR(TF_UINT32, TF_FLOAT, shader::Cast_uint_float)
#endif
#ifdef CAST_UINT_INT
  REGISTER_TYPE_PAIR(TF_UINT32, TF_INT32, shader::Cast_uint_int)
#endif
#ifdef CAST_UINT_INT64_T
  REGISTER_TYPE_PAIR(TF_UINT32, TF_INT64, shader::Cast_uint_int64_t)
#endif
#ifdef CAST_UINT_UINT64_T
  REGISTER_TYPE_PAIR(TF_UINT32, TF_UINT64, shader::Cast_uint_uint64_t)
#endif
#ifdef CAST_UINT_INT8_T
  REGISTER_TYPE_PAIR(TF_UINT32, TF_INT8, shader::Cast_uint_int8_t)
#endif
#ifdef CAST_UINT_UINT8_T
  REGISTER_TYPE_PAIR(TF_UINT32, TF_UINT8, shader::Cast_uint_uint8_t)
#endif
#ifdef CAST_UINT_DOUBLE
  REGISTER_TYPE_PAIR(TF_UINT32, TF_DOUBLE, shader::Cast_uint_double)
#endif

#ifdef CAST_INT64_T_INT64_T
  REGISTER_TYPE_PAIR(TF_INT64, TF_INT64, shader::Cast_int64_t_int64_t)
#endif
#ifdef CAST_INT64_T_INT
  REGISTER_TYPE_PAIR(TF_INT64, TF_INT32, shader::Cast_int64_t_int)
#endif
#ifdef CAST_INT64_T_UINT
  REGISTER_TYPE_PAIR(TF_INT64, TF_UINT32, shader::Cast_int64_t_uint)
#endif
#ifdef CAST_INT64_T_FLOAT
  REGISTER_TYPE_PAIR(TF_INT64, TF_FLOAT, shader::Cast_int64_t_float)
#endif
#ifdef CAST_INT64_T_UINT64_T
  REGISTER_TYPE_PAIR(TF_INT64, TF_UINT64, shader::Cast_int64_t_uint64_t)
#endif
#ifdef CAST_INT64_T_INT8_T
  REGISTER_TYPE_PAIR(TF_INT64, TF_INT8, shader::Cast_int64_t_int8_t)
#endif
#ifdef CAST_INT64_T_UINT8_T
  REGISTER_TYPE_PAIR(TF_INT64, TF_UINT8, shader::Cast_int64_t_uint8_t)
#endif
#ifdef CAST_INT64_T_DOUBLE
  REGISTER_TYPE_PAIR(TF_INT64, TF_DOUBLE, shader::Cast_int64_t_double)
#endif

#ifdef CAST_UINT64_T_UINT64_T
  REGISTER_TYPE_PAIR(TF_UINT64, TF_UINT64, shader::Cast_uint64_t_uint64_t)
#endif
#ifdef CAST_UINT64_T_INT
  REGISTER_TYPE_PAIR(TF_UINT64, TF_INT32, shader::Cast_uint64_t_int)
#endif
#ifdef CAST_UINT64_T_UINT
  REGISTER_TYPE_PAIR(TF_UINT64, TF_UINT32, shader::Cast_uint64_t_uint)
#endif
#ifdef CAST_UINT64_T_FLOAT
  REGISTER_TYPE_PAIR(TF_UINT64, TF_FLOAT, shader::Cast_uint64_t_float)
#endif
#ifdef CAST_UINT64_T_INT64_T
  REGISTER_TYPE_PAIR(TF_UINT64, TF_INT64, shader::Cast_uint64_t_int64_t)
#endif
#ifdef CAST_UINT64_T_INT8_T
  REGISTER_TYPE_PAIR(TF_UINT64, TF_INT8, shader::Cast_uint64_t_int8_t)
#endif
#ifdef CAST_UINT64_T_UINT8_T
  REGISTER_TYPE_PAIR(TF_UINT64, TF_UINT8, shader::Cast_uint64_t_uint8_t)
#endif
#ifdef CAST_UINT64_T_DOUBLE
  REGISTER_TYPE_PAIR(TF_UINT64, TF_DOUBLE, shader::Cast_uint64_t_double)
#endif

#ifdef CAST_INT8_T_INT8_T
  REGISTER_TYPE_PAIR(TF_INT8, TF_INT8, shader::Cast_int8_t_int8_t)
#endif
#ifdef CAST_INT8_T_FLOAT
  REGISTER_TYPE_PAIR(TF_INT8, TF_FLOAT, shader::Cast_int8_t_float)
#endif
#ifdef CAST_INT8_T_INT
  REGISTER_TYPE_PAIR(TF_INT8, TF_INT32, shader::Cast_int8_t_int)
#endif
#ifdef CAST_INT8_T_UINT
  REGISTER_TYPE_PAIR(TF_INT8, TF_UINT32, shader::Cast_int8_t_uint)
#endif
#ifdef CAST_INT8_T_INT64_T
  REGISTER_TYPE_PAIR(TF_INT8, TF_INT64, shader::Cast_int8_t_int64_t)
#endif
#ifdef CAST_INT8_T_UINT64_T
  REGISTER_TYPE_PAIR(TF_INT8, TF_UINT64, shader::Cast_int8_t_uint64_t)
#endif
#ifdef CAST_INT8_T_UINT8_T
  REGISTER_TYPE_PAIR(TF_INT8, TF_UINT8, shader::Cast_int8_t_uint8_t)
#endif
#ifdef CAST_INT8_T_DOUBLE
  REGISTER_TYPE_PAIR(TF_INT8, TF_DOUBLE, shader::Cast_int8_t_double)
#endif

#ifdef CAST_UINT8_T_UINT8_T
  REGISTER_TYPE_PAIR(TF_UINT8, TF_UINT8, shader::Cast_uint8_t_uint8_t)
#endif
#ifdef CAST_UINT8_T_FLOAT
  REGISTER_TYPE_PAIR(TF_UINT8, TF_FLOAT, shader::Cast_uint8_t_float)
#endif
#ifdef CAST_UINT8_T_INT
  REGISTER_TYPE_PAIR(TF_UINT8, TF_INT32, shader::Cast_uint8_t_int)
#endif
#ifdef CAST_UINT8_T_UINT
  REGISTER_TYPE_PAIR(TF_UINT8, TF_UINT32, shader::Cast_uint8_t_uint)
#endif
#ifdef CAST_UINT8_T_INT64_T
  REGISTER_TYPE_PAIR(TF_UINT8, TF_INT64, shader::Cast_uint8_t_int64_t)
#endif
#ifdef CAST_UINT8_T_UINT64_T
  REGISTER_TYPE_PAIR(TF_UINT8, TF_UINT64, shader::Cast_uint8_t_uint64_t)
#endif
#ifdef CAST_UINT8_T_INT8_T
  REGISTER_TYPE_PAIR(TF_UINT8, TF_INT8, shader::Cast_uint8_t_int8_t)
#endif
#ifdef CAST_UINT8_T_DOUBLE
  REGISTER_TYPE_PAIR(TF_UINT8, TF_DOUBLE, shader::Cast_uint8_t_double)
#endif

#ifdef CAST_DOUBLE_DOUBLE
  REGISTER_TYPE_PAIR(TF_DOUBLE, TF_DOUBLE, shader::Cast_double_double)
#endif
#ifdef CAST_DOUBLE_FLOAT
  REGISTER_TYPE_PAIR(TF_DOUBLE, TF_FLOAT, shader::Cast_double_float)
#endif
#ifdef CAST_DOUBLE_INT
  REGISTER_TYPE_PAIR(TF_DOUBLE, TF_INT32, shader::Cast_double_int)
#endif
#ifdef CAST_DOUBLE_UINT
  REGISTER_TYPE_PAIR(TF_DOUBLE, TF_UINT32, shader::Cast_double_uint)
#endif
#ifdef CAST_DOUBLE_INT64_T
  REGISTER_TYPE_PAIR(TF_DOUBLE, TF_INT64, shader::Cast_double_int64_t)
#endif
#ifdef CAST_DOUBLE_UINT64_T
  REGISTER_TYPE_PAIR(TF_DOUBLE, TF_UINT64, shader::Cast_double_uint64_t)
#endif
#ifdef CAST_DOUBLE_INT8_T
  REGISTER_TYPE_PAIR(TF_DOUBLE, TF_INT8, shader::Cast_double_int8_t)
#endif
#ifdef CAST_DOUBLE_UINT8_T
  REGISTER_TYPE_PAIR(TF_DOUBLE, TF_UINT8, shader::Cast_double_uint8_t)
#endif
}