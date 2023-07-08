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

  tensor_utills::Input_tensor input =
      tensor_utills::get_input_tensor("SumOp:input", 0, ctx, status.get());

  if (input.is_empty) {
    absl::InlinedVector<int64_t, 4> out_dims(0);
    tensor_utills::Output_tensor output = tensor_utills::make_output_tensor(
        "SumOp:output", 0, out_dims, T, ctx, status.get());

    SP_Stream stream = TF_GetStream(ctx, status.get());
    vulten_backend::Instance* inst = stream->instance;
    inst->fill_buffer(output.vulten_tensor.buffer, 0, VK_WHOLE_SIZE, 0);
    return;
  }
  if (input.is_scalar) {
    tensor_utills::Output_tensor output = tensor_utills::make_output_tensor(
        "SumOp:output", 0, input.dims, T, ctx, status.get());

    SP_Stream stream = TF_GetStream(ctx, status.get());
    vulten_backend::Instance* inst = stream->instance;
    inst->copy_buffer(input.vulten_tensor.buffer, output.vulten_tensor.buffer);
    return;
  }

  tensor_utills::Input_host_tensor axis =
      tensor_utills::get_input_host_tensor("SumOp:axis", 1, ctx, status.get());

  if (axis.is_scalar) {
    axis.dims.resize(1, 1);
  }

  std::vector<int32_t> axis_vec = std::vector<int32_t>(axis.dims[0]);

  // For now we convert int64 axis tensors to int32, but we could make the
  // shader accept both if need be.
  switch (axis.type) {
    case TF_INT32:
      memcpy(axis_vec.data(), axis.data, sizeof(int32_t) * axis_vec.size());
      break;
    case TF_INT64:
      for (uint32_t i = 0; i < axis_vec.size(); i++) {
        axis_vec[i] = uint32_t(((int64_t*)axis.data)[i]);
      }
      break;
    default:
      std::cerr << "Error invalid axis tensor proved to SumOp\n";
      exit(-1);
  }

  for (uint32_t i = 0; i < axis_vec.size(); i++) {
    if (axis_vec[i] < 0) {
      axis_vec[i] += int32_t(input.dims.size());
    }
  }

  if (!axis.is_scalar) {
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
  }

  absl::InlinedVector<int64_t, 4> out_dims = input.dims;
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

  tensor_utills::Output_tensor output = tensor_utills::make_output_tensor(
      "SumOp:output", 0, out_dims, T, ctx, status.get());

  SP_Stream stream = TF_GetStream(ctx, status.get());
  vulten_backend::Instance* inst = stream->instance;

  if (axis_vec.size() == 0) {
    inst->copy_buffer(input.vulten_tensor.buffer, output.vulten_tensor.buffer);
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

  sum_op->run_op((vulten_ops::Data_type)T, input.vulten_tensor, axis_vec,
                 output.vulten_tensor);
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