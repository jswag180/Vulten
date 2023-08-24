#include "Vulten_backend_ops.h"

#include <filesystem>
#include <fstream>

namespace vulten_ops {

std::string Data_type_to_str(Data_type dt) {
  if (dt == Data_type::VULTEN_FLOAT) {
    return "float";
  } else if (dt == Data_type::VULTEN_FLOAT16) {
    return "float16_t";
  } else if (dt == Data_type::VULTEN_DOUBLE) {
    return "double";
  } else if (dt == Data_type::VULTEN_INT32) {
    return "int";
  } else if (dt == Data_type::VULTEN_UINT32) {
    return "uint";
  } else if (dt == Data_type::VULTEN_INT64) {
    return "int64_t";
  } else if (dt == Data_type::VULTEN_UINT64) {
    return "uint64_t";
  } else if (dt == Data_type::VULTEN_INT8) {
    return "int8_t";
  } else if (dt == Data_type::VULTEN_UINT8) {
    return "uint8_t";
  } else if (dt == Data_type::VULTEN_INT16) {
    return "int16_t";
  } else if (dt == Data_type::VULTEN_UINT16) {
    return "uint16_t";
  } else if (dt == Data_type::VULTEN_COMPLEX64) {
    return "cx_64";
  } else if (dt == Data_type::VULTEN_COMPLEX128) {
    return "cx_128";
  } else if (dt == Data_type::VULTEN_BOOL) {
    return "bool8";
  } else {
    throw std::runtime_error(
        "Error not a valid vulten_ops::DataType passed to "
        "Data_type_to_str(Data_type dt).");
  }
}

Vulten_tensor::Vulten_tensor(vulten_backend::Buffer* buffer_ptr,
                             int64_t num_dims, int64_t* dims_ptr)
    : buffer(buffer_ptr), num_dims(num_dims), dims(dims_ptr) {
  //
}

int64_t Vulten_tensor::get_total_elements() {
  int64_t total_elements = 1;
  for (int64_t i = 0; i < num_dims; i++) {
    total_elements *= dims[i];
  }
  return total_elements;
}

Vulten_tensor::~Vulten_tensor() {
  //
}

}  // namespace vulten_ops