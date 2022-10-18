#include "absl/container/inlined_vector.h"
#include "gpuBackend.h"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "vulten_device.h"

// shaders
// float  float
// int    int
// uint   uint

// float int 1
// float uint 2
// int float 3
// int uint 4
// uint float 5
// uint int 6
#include <chrono>
#include <iostream>
#include <map>
#include <memory>
#include <vector>

#include "shaders/headers/shaderCast_1.hpp"
#include "shaders/headers/shaderCast_10.hpp"
#include "shaders/headers/shaderCast_2.hpp"
#include "shaders/headers/shaderCast_3.hpp"
#include "shaders/headers/shaderCast_4.hpp"
#include "shaders/headers/shaderCast_5.hpp"
#include "shaders/headers/shaderCast_6.hpp"
#include "shaders/headers/shaderCast_7.hpp"
#include "shaders/headers/shaderCast_8.hpp"
#include "shaders/headers/shaderCast_9.hpp"

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
static std::vector<uint32_t> spirv_cast_1;
static std::vector<uint32_t> spirv_cast_2;
static std::vector<uint32_t> spirv_cast_3;
static std::vector<uint32_t> spirv_cast_4;
static std::vector<uint32_t> spirv_cast_5;
static std::vector<uint32_t> spirv_cast_6;
static std::vector<uint32_t> spirv_cast_7;
static std::vector<uint32_t> spirv_cast_8;
static std::vector<uint32_t> spirv_cast_9;
static std::vector<uint32_t> spirv_cast_10;
static std::map<std::pair<TF_DataType, TF_DataType>, std::vector<uint32_t>*>
    shaders;

std::string makeTypePair(TF_DataType source, TF_DataType destination) {
  return source + "," + destination;
}

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

template <TF_DataType S, TF_DataType D>
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
      {*in_ptr, *out_ptr}, *shaders[{S, D}],
      kp::Workgroup({in_ptr->get()->size()}), std::vector<uint32_t>{D}, {});

  stream->instance->mngr->sequence(stream->instance->mainQueue)
      ->record<kp::OpAlgoDispatch>(algo)
      ->eval();
}

template <TF_DataType S, TF_DataType D>
void RegisterCastOpKernel(const char* device_type) {
  StatusSafePtr status(TF_NewStatus());
  auto* builder = TF_NewKernelBuilder("Cast", device_type, CastOp_Create<S, D>,
                                      &CastOp_Compute<S, D>, &CastOp_Delete);

  TF_KernelBuilder_TypeConstraint(builder, "SrcT", S, status.get());
  TF_KernelBuilder_TypeConstraint(builder, "DstT", D, status.get());

  if (TF_OK != TF_GetCode(status.get()))
    std::cout << " Error while registering Cast kernel with attribute T";
  TF_RegisterKernelBuilder("Cast", builder, status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout << " Error while registering Cast kernel";
}

}  // namespace vulten_plugin

#define REGISTER_TYPE_PAIR(s, d, c)    \
  vulten_plugin::shaders[{s, d}] = &c; \
  vulten_plugin::RegisterCastOpKernel<s, d>(device_type);

void RegisterDeviceCast(const char* device_type) {
  LOAD_SHADER_TO_VEC(vulten_plugin::spirv_cast_1,
                     kp::shader_data::___shaders_cast_Cast_1_comp_spv)
  LOAD_SHADER_TO_VEC(vulten_plugin::spirv_cast_2,
                     kp::shader_data::___shaders_cast_Cast_2_comp_spv)
  LOAD_SHADER_TO_VEC(vulten_plugin::spirv_cast_3,
                     kp::shader_data::___shaders_cast_Cast_3_comp_spv)
  LOAD_SHADER_TO_VEC(vulten_plugin::spirv_cast_4,
                     kp::shader_data::___shaders_cast_Cast_4_comp_spv)
  LOAD_SHADER_TO_VEC(vulten_plugin::spirv_cast_5,
                     kp::shader_data::___shaders_cast_Cast_5_comp_spv)
  LOAD_SHADER_TO_VEC(vulten_plugin::spirv_cast_6,
                     kp::shader_data::___shaders_cast_Cast_6_comp_spv)
  LOAD_SHADER_TO_VEC(vulten_plugin::spirv_cast_7,
                     kp::shader_data::___shaders_cast_Cast_7_comp_spv)
  LOAD_SHADER_TO_VEC(vulten_plugin::spirv_cast_8,
                     kp::shader_data::___shaders_cast_Cast_8_comp_spv)
  LOAD_SHADER_TO_VEC(vulten_plugin::spirv_cast_9,
                     kp::shader_data::___shaders_cast_Cast_9_comp_spv)
  LOAD_SHADER_TO_VEC(vulten_plugin::spirv_cast_10,
                     kp::shader_data::___shaders_cast_Cast_10_comp_spv)

  // float | int uint       1
  // float | int64 uint64   2
  // int   | float uint     3
  // int   | int64 uint64   4
  // uint  | float int      5
  // uint  | int64 uint64   6
  // int64 | float int uint 7
  // int64 | uint64         8
  // uint64| float int uint 9
  // uint64| int64          10

  REGISTER_TYPE_PAIR(TF_FLOAT, TF_INT32, vulten_plugin::spirv_cast_1)
  REGISTER_TYPE_PAIR(TF_FLOAT, TF_UINT32, vulten_plugin::spirv_cast_1)

  REGISTER_TYPE_PAIR(TF_FLOAT, TF_INT64, vulten_plugin::spirv_cast_2)
  REGISTER_TYPE_PAIR(TF_FLOAT, TF_UINT64, vulten_plugin::spirv_cast_2)

  REGISTER_TYPE_PAIR(TF_INT32, TF_FLOAT, vulten_plugin::spirv_cast_3)
  REGISTER_TYPE_PAIR(TF_INT32, TF_UINT32, vulten_plugin::spirv_cast_3)

  REGISTER_TYPE_PAIR(TF_INT32, TF_INT64, vulten_plugin::spirv_cast_4)
  REGISTER_TYPE_PAIR(TF_INT32, TF_UINT64, vulten_plugin::spirv_cast_4)

  REGISTER_TYPE_PAIR(TF_UINT32, TF_FLOAT, vulten_plugin::spirv_cast_5)
  REGISTER_TYPE_PAIR(TF_UINT32, TF_INT32, vulten_plugin::spirv_cast_5)

  REGISTER_TYPE_PAIR(TF_UINT32, TF_INT64, vulten_plugin::spirv_cast_6)
  REGISTER_TYPE_PAIR(TF_UINT32, TF_UINT64, vulten_plugin::spirv_cast_6)

  REGISTER_TYPE_PAIR(TF_INT64, TF_FLOAT, vulten_plugin::spirv_cast_7)
  REGISTER_TYPE_PAIR(TF_INT64, TF_INT32, vulten_plugin::spirv_cast_7)
  REGISTER_TYPE_PAIR(TF_INT64, TF_UINT32, vulten_plugin::spirv_cast_7)

  REGISTER_TYPE_PAIR(TF_INT64, TF_UINT64, vulten_plugin::spirv_cast_8)

  REGISTER_TYPE_PAIR(TF_UINT64, TF_FLOAT, vulten_plugin::spirv_cast_9)
  REGISTER_TYPE_PAIR(TF_UINT64, TF_INT32, vulten_plugin::spirv_cast_9)
  REGISTER_TYPE_PAIR(TF_UINT64, TF_UINT32, vulten_plugin::spirv_cast_9)

  REGISTER_TYPE_PAIR(TF_UINT64, TF_INT64, vulten_plugin::spirv_cast_10)
}