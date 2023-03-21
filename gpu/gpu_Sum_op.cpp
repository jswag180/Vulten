#include "Vulten_backend/ops/Sum_op.h"
#include "absl/container/inlined_vector.h"
#include "scope_timer.h"
#include "tensor_utills.h"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "vulten_device.h"

struct SumOp {
  SumOp() : keep_dims_(false) {}
  bool keep_dims_;
};

void* SumOp_Create(TF_OpKernelConstruction* ctx) {
  auto kernel = new SumOp();

  StatusSafePtr status(TF_NewStatus());

  TF_OpKernelConstruction_GetAttrBool(
      ctx, "keep_dims", (unsigned char*)&kernel->keep_dims_, status.get());

  return kernel;
}

void SumOp_Delete(void* kernel) {
  if (kernel != nullptr) {
    delete static_cast<SumOp*>(kernel);
  }
}

template <TF_DataType T>
void SumOp_Compute(void* kernel, TF_OpKernelContext* ctx) {
  SCOPE_TIMER("SumOp")

  StatusSafePtr status(TF_NewStatus());

  SumOp* sumOp_info = static_cast<SumOp*>(kernel);

  GET_INPUT_TENSOR("Sum", input, 0, ctx, status)

  // This input is a host buffer
  TF_Tensor* axis = nullptr;
  TF_GetInput(ctx, 1, &axis, status.get());
  TensorSafePtr axis_safe_ptr(axis);
  if (TF_GetCode(status.get()) != TF_OK) {
    TF_OpKernelContext_Failure(ctx, status.get());
    std::cout << "Error: "
              << "axis"
              << " at " << axis << "\n";
    return;
  }
  absl::InlinedVector<int64_t, 4> axis_dims(TF_NumDims(axis_safe_ptr.get()));
  for (auto i = 0; i < TF_NumDims(axis_safe_ptr.get()); ++i) {
    axis_dims[i] = TF_Dim(axis_safe_ptr.get(), i);
  }

  std::vector<int32_t> axis_vec = std::vector<int32_t>(axis_dims[0]);
  if (TF_TensorType(axis_safe_ptr.get()) == TF_INT32) {
    int32_t* axis_data = (int32_t*)TF_TensorData(axis_safe_ptr.get());
    memcpy(axis_vec.data(), axis_data, sizeof(int32_t) * axis_vec.size());
  } else if (TF_TensorType(axis_safe_ptr.get()) == TF_INT64) {
    int64_t* axis_data = (int64_t*)TF_TensorData(axis_safe_ptr.get());
    for (uint32_t i = 0; i < axis_vec.size(); i++) {
      axis_vec[i] = uint32_t(axis_data[i]);
    }
  }

  for (uint32_t i = 0; i < axis_vec.size(); i++) {
    if (axis_vec[i] < 0) {
      axis_vec[i] += int32_t(input_dims.size());
    }
  }

  // sort from low to high
  int32_t i, key, j;
  for (i = 1; i < axis_vec.size(); i++) {
    key = axis_vec[i];
    j = i - 1;

    while (j >= 0 && axis_vec[j] > key) {
      axis_vec[j + 1] = axis_vec[j];
      j = j - 1;
    }
    axis_vec[j + 1] = key;
  }

  absl::InlinedVector<int64_t, 4> out_dims = input_dims;
  if (axis_vec.size() != 0) {
    if (sumOp_info->keep_dims_) {
      for (uint32_t i = 0; i < axis_vec.size(); i++) {
        out_dims[axis_vec[i]] = 1;
      }
    } else {
      for (int32_t i = axis_vec.size() - 1; i > -1; i--) {
        out_dims.erase(out_dims.begin() + axis_vec[i]);
      }
    }
  }

  MAKE_OUTPUT_TENSOR("Sum", output, 0, out_dims, T, ctx, status)

  SP_Stream stream = TF_GetStream(ctx, status.get());
  vulten_backend::Instance* inst = stream->instance;

  if (axis_vec.size() == 0) {
    inst->copy_buffer(input_tensor.buffer, output_tensor.buffer);
    return;
  }

  vulten_ops::Sum_op* sum_op = nullptr;
  std::string op_cache_name = "Sum";
  inst->main_queue_mutex.lock();
  if (inst->op_chache.find(op_cache_name) == inst->op_chache.end()) {
    inst->op_chache[op_cache_name] =
        (vulten_ops::Vulten_op*)new vulten_ops::Sum_op(inst);
  }
  sum_op = (vulten_ops::Sum_op*)inst->op_chache[op_cache_name];
  inst->main_queue_mutex.unlock();

  std::reverse(axis_vec.begin(), axis_vec.end());

  sum_op->run_op((vulten_ops::Data_type)T, input_tensor, axis_vec,
                 output_tensor);
}

template <TF_DataType T>
void RegisterSumOpKernel(const char* device_type) {
  StatusSafePtr status(TF_NewStatus());
  auto* builder = TF_NewKernelBuilder("Sum", device_type, SumOp_Create,
                                      &SumOp_Compute<T>, &SumOp_Delete);
  TF_KernelBuilder_TypeConstraint(builder, "T", T, status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout << " Error while registering Sum kernel with attribute T";

  TF_KernelBuilder_HostMemory(builder, "reduction_indices");

  TF_RegisterKernelBuilder("Sum", builder, status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout << " Error while registering Sum kernel";
}

void RegisterDeviceSum(const char* device_type) {
#define REGISTER_KERNEL(T) RegisterSumOpKernel<T>(device_type);

  CALL_ALL_TYPES(REGISTER_KERNEL)
}