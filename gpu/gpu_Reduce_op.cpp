#include "Vulten_backend/ops/reduce/Reduce_op.h"
#include "absl/container/inlined_vector.h"
#include "scope_timer.h"
#include "tensor_utills.h"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "vulten_device.h"

struct ReduceOp {
  ReduceOp() : keep_dims_(false) {}
  bool keep_dims_;
};

void* ReduceOp_Create(TF_OpKernelConstruction* ctx) {
  auto kernel = new ReduceOp();

  StatusSafePtr status(TF_NewStatus());

  TF_OpKernelConstruction_GetAttrBool(
      ctx, "keep_dims", (unsigned char*)&kernel->keep_dims_, status.get());

  return kernel;
}

void ReduceOp_Delete(void* kernel) {
  if (kernel != nullptr) {
    delete static_cast<ReduceOp*>(kernel);
  }
}

template <TF_DataType T, uint32_t OP>
void ReduceOp_Compute(void* kernel, TF_OpKernelContext* ctx) {
  SCOPE_TIMER("ReduceOp")

  StatusSafePtr status(TF_NewStatus());

  ReduceOp* reduceOp_info = static_cast<ReduceOp*>(kernel);

  tensor_utills::Input_tensor input =
      tensor_utills::get_input_tensor("ReduceOp:input", 0, ctx, status.get());

  if (input.is_empty) {
    absl::InlinedVector<int64_t, 4> out_dims(0);
    tensor_utills::Output_tensor output = tensor_utills::make_output_tensor(
        "ReduceOp:output", 0, out_dims, ctx, status.get());

    SP_Stream stream = TF_GetStream(ctx, status.get());
    vulten_backend::Instance* inst = stream->instance;
    inst->fill_buffer(output.vulten_tensor.buffer, 0, VK_WHOLE_SIZE, 0);
    return;
  }
  if (input.is_scalar) {
    tensor_utills::Output_tensor output = tensor_utills::make_output_tensor(
        "ReduceOp:output", 0, input.dims, ctx, status.get());

    SP_Stream stream = TF_GetStream(ctx, status.get());
    vulten_backend::Instance* inst = stream->instance;
    inst->copy_buffer(input.vulten_tensor.buffer, output.vulten_tensor.buffer);
    return;
  }

  tensor_utills::Input_host_tensor axis = tensor_utills::get_input_host_tensor(
      "ReduceOp:axis", 1, ctx, status.get());

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
      std::cerr << "Error invalid axis tensor proved to ReduceOp\n";
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
    if (reduceOp_info->keep_dims_) {
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
      "ReduceOp:output", 0, out_dims, ctx, status.get());

  SP_Stream stream = TF_GetStream(ctx, status.get());
  vulten_backend::Instance* inst = stream->instance;

  if (axis_vec.size() == 0) {
    inst->copy_buffer(input.vulten_tensor.buffer, output.vulten_tensor.buffer);
    return;
  }

  std::reverse(axis_vec.begin(), axis_vec.end());

  vulten_ops::reduce::run_op(inst, (vulten_ops::Data_type)T,
                             input.vulten_tensor, axis_vec,
                             output.vulten_tensor, OP);
}

template <TF_DataType T, uint32_t OP>
void RegisterReduceOpKernel(const char* device_type) {
  std::string op = vulten_ops::reduce::op_as_str(OP);

  StatusSafePtr status(TF_NewStatus());
  auto* builder =
      TF_NewKernelBuilder(op.c_str(), device_type, ReduceOp_Create,
                          &ReduceOp_Compute<T, OP>, &ReduceOp_Delete);
  TF_KernelBuilder_TypeConstraint(builder, "T", T, status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout << " Error while registering Reduce kernel with attribute T";

  TF_KernelBuilder_HostMemory(builder, "reduction_indices");

  TF_RegisterKernelBuilder(op.c_str(), builder, status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout << " Error while registering Reduce kernel";
}

void RegisterDeviceReduce(const char* device_type) {
#define REGISTER_SUM_KERNEL(T) RegisterReduceOpKernel<T, OP_SUM>(device_type);
#define REGISTER_MAX_MIN_KERNEL(T)                \
  RegisterReduceOpKernel<T, OP_MAX>(device_type); \
  RegisterReduceOpKernel<T, OP_MIN>(device_type);

  CALL_ALL_TYPES(REGISTER_SUM_KERNEL)
  CALL_ALL_BASIC_TYPES(REGISTER_MAX_MIN_KERNEL)
}