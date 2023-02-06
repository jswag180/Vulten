#include "Vulten_backend/ops/Basic_ops.h"
#include "absl/container/inlined_vector.h"
#include "scope_timer.h"
#include "tensor_utills.h"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "vulten_device.h"

template <TF_DataType T, uint32_t OP>
void BasicOps_Compute(void* kernel, TF_OpKernelContext* ctx) {
  SCOPE_TIMER("BasicOp " + vulten_ops::op_as_str(OP))

  StatusSafePtr status(TF_NewStatus());

  TF_Tensor* x = nullptr;
  TF_GetInput(ctx, 0, &x, status.get());
  TensorSafePtr x_safe_ptr(x);
  if (TF_GetCode(status.get()) != TF_OK) {
    TF_OpKernelContext_Failure(ctx, status.get());
    std::cout << "Error: Basic x input\n";
    return;
  }
  if (TF_TensorElementCount(x_safe_ptr.get()) == 0) return;
  auto x_ptr = VOID_TO_DEVICE_BUFFER(TF_TensorData(x_safe_ptr.get()));

  TF_Tensor* y = nullptr;
  TF_GetInput(ctx, 1, &y, status.get());
  TensorSafePtr y_safe_ptr(y);
  if (TF_GetCode(status.get()) != TF_OK) {
    TF_OpKernelContext_Failure(ctx, status.get());
    std::cout << "Error: Basic y input\n";
    return;
  }
  if (TF_TensorElementCount(y_safe_ptr.get()) == 0) return;
  auto y_ptr = VOID_TO_DEVICE_BUFFER(TF_TensorData(y_safe_ptr.get()));

  absl::InlinedVector<int64_t, 4> x_dims =
      absl::InlinedVector<int64_t, 4>(4, 1);
  for (auto i = 0; i < TF_NumDims(x_safe_ptr.get()); ++i) {
    x_dims[i + (4 - TF_NumDims(x_safe_ptr.get()))] =
        TF_Dim(x_safe_ptr.get(), i);
  }

  absl::InlinedVector<int64_t, 4> y_dims =
      absl::InlinedVector<int64_t, 4>(4, 1);
  for (auto i = 0; i < TF_NumDims(y_safe_ptr.get()); ++i) {
    y_dims[i + (4 - TF_NumDims(y_safe_ptr.get()))] =
        TF_Dim(y_safe_ptr.get(), i);
  }

  absl::InlinedVector<int64_t, 4> res_dims =
      absl::InlinedVector<int64_t, 4>(4, 1);
  int64_t resNumElements = 1;
  int64_t numOfResDims =
      std::max(TF_NumDims(x_safe_ptr.get()), TF_NumDims(y_safe_ptr.get()));
  for (int64_t i = 3; i >= 0; i--) {
    res_dims[i] = std::max(x_dims[i], y_dims[i]);
    resNumElements *= res_dims[i];
  }

  TensorSafePtr output_safe_ptr(
      TF_AllocateOutput(ctx, 0, TF_ExpectedOutputDataType(ctx, 0),
                        &res_dims.data()[4 - numOfResDims], numOfResDims,
                        resNumElements * TF_DataTypeSize(T), status.get()));
  if (TF_GetCode(status.get()) != TF_OK) {
    TF_OpKernelContext_Failure(ctx, status.get());
    std::cout << "Error: Basic output\n";
    return;
  }
  auto out_ptr = VOID_TO_DEVICE_BUFFER(TF_TensorData(output_safe_ptr.get()));

  vulten_ops::Vulten_tensor x_tensor(x_ptr, x_dims.size(), x_dims.data());
  vulten_ops::Vulten_tensor y_tensor(y_ptr, y_dims.size(), y_dims.data());
  vulten_ops::Vulten_tensor output_tensor(out_ptr, res_dims.size(),
                                          res_dims.data());

  SP_Stream stream = TF_GetStream(ctx, status.get());
  vulten_backend::Instance* inst = stream->instance;

  vulten_ops::Basic_op* basic_op = nullptr;
  std::string op_cache_name = "Basic";
  inst->main_queue_mutex.lock();
  if (inst->op_chache.find(op_cache_name) == inst->op_chache.end()) {
    inst->op_chache[op_cache_name] =
        (vulten_ops::Vulten_op*)new vulten_ops::Basic_op(inst);
  }
  basic_op = (vulten_ops::Basic_op*)inst->op_chache[op_cache_name];
  inst->main_queue_mutex.unlock();

  basic_op->run_op((vulten_ops::Data_type)T, OP, x_tensor, y_tensor,
                   output_tensor);
}

template <TF_DataType T, uint32_t OP>
void RegisterBasicOpKernels(const char* device_type) {
  std::string op = vulten_ops::op_as_str(OP);

  StatusSafePtr status(TF_NewStatus());
  auto* builder = TF_NewKernelBuilder(op.c_str(), device_type, nullptr,
                                      &BasicOps_Compute<T, OP>, nullptr);
  TF_KernelBuilder_TypeConstraint(builder, "T", T, status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout << " Error while registering Basic kernel with attribute T";
  TF_RegisterKernelBuilder(op.c_str(), builder, status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout << " Error while registering Basic kernel";

  // There is no apparent difference between Add and AddV2 so use the same for
  // both
  if (OP == OP_ADD) {
    StatusSafePtr status(TF_NewStatus());
    auto* builder = TF_NewKernelBuilder("AddV2", device_type, nullptr,
                                        &BasicOps_Compute<T, OP>, nullptr);
    TF_KernelBuilder_TypeConstraint(builder, "T", T, status.get());
    if (TF_OK != TF_GetCode(status.get()))
      std::cout << " Error while registering Basic kernel with attribute T";
    TF_RegisterKernelBuilder("AddV2", builder, status.get());
    if (TF_OK != TF_GetCode(status.get()))
      std::cout << " Error while registering Basic kernel";
  }
}

void RegisterDeviceBasicOps(const char* device_type) {
#define REGISTER_KERNEL(T)                        \
  RegisterBasicOpKernels<T, OP_MUL>(device_type); \
  RegisterBasicOpKernels<T, OP_ADD>(device_type); \
  RegisterBasicOpKernels<T, OP_SUB>(device_type); \
  RegisterBasicOpKernels<T, OP_DIV>(device_type); \
  RegisterBasicOpKernels<T, OP_DIV_NO_NAN>(device_type);

  CALL_ALL_BASIC_TYPES(REGISTER_KERNEL)

#define REGISTER_COMPLEX_KERNEL(T)                \
  RegisterBasicOpKernels<T, OP_ADD>(device_type); \
  RegisterBasicOpKernels<T, OP_SUB>(device_type); \
  RegisterBasicOpKernels<T, OP_MUL>(device_type); \
  RegisterBasicOpKernels<T, OP_DIV>(device_type);

  CALL_COMPLEX(REGISTER_COMPLEX_KERNEL)
}