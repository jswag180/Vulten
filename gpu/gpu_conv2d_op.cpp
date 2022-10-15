#include <math.h>
#include <string.h>

#include <iostream>
#include <memory>
#include <vector>

#include "gpuBackend.h"
#include "shaders/headers/shaderconv2d.hpp"
#include "shaders/headers/shaderconv2dOld.hpp"
#include "shaders/headers/shaderim2colSame.hpp"
#include "shaders/headers/shaderim2colValid.hpp"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_tensor.h"
#include "vulten_device.h"

struct StatusDeleter {
  void operator()(TF_Status *s) {
    if (s != nullptr) {
      TF_DeleteStatus(s);
    }
  }
};

struct TensorDeleter {
  void operator()(TF_Tensor *t) {
    if (t != nullptr) {
      TF_DeleteTensor(t);
    }
  }
};

using StatusSafePtr = std::unique_ptr<TF_Status, StatusDeleter>;
using TensorSafePtr = std::unique_ptr<TF_Tensor, TensorDeleter>;

enum Padding {
  VALID = 1,     // No padding.
  SAME = 2,      // Input and output layers have the same size.
  EXPLICIT = 3,  // Padding is explicitly specified.
};

enum TensorFormat {
  FORMAT_NHWC = 0,
  FORMAT_NCHW = 1,
  FORMAT_NCHW_VECT_C = 2,
  FORMAT_NHWC_VECT_W = 3,
  FORMAT_HWNC = 4,
  FORMAT_HWCN = 5,
};

static bool GetWindowedOutputSize(int64_t input_size, int64_t filter_size,
                                  int64_t dilation_rate, int64_t stride,
                                  Padding padding_type, int64_t *output_size,
                                  int64_t *padding_before) {
  if (stride <= 0) {
    std::cerr << "Stride must be > 0, but got " << stride << std::endl;
    return false;
  }
  if (dilation_rate < 1) {
    std::cerr << "Dilation rate must be >= 1, but got " << dilation_rate
              << std::endl;
    return false;
  }

  int64_t effective_filter_size = (filter_size - 1) * dilation_rate + 1;
  switch (padding_type) {
    case Padding::VALID:
      *output_size = (input_size - effective_filter_size + stride) / stride;
      *padding_before = 0;
      break;
    case Padding::SAME:
      *output_size = (input_size + stride - 1) / stride;
      const int64_t padding_needed =
          std::max(int64_t{0}, (*output_size - 1) * stride +
                                   effective_filter_size - input_size);
      // For odd values of total padding, add more padding at the 'right'
      // side of the given dimension.
      *padding_before = padding_needed / 2;
      break;
  }
  if (*output_size < 0) {
    std::cerr << "Computed output size would be negative: " << *output_size
              << " [input_size: " << input_size
              << ", effective_filter_size: " << effective_filter_size
              << ", stride: " << stride << "]" << std::endl;
    return false;
  }
  return true;
}

static int64_t GetTensorDim(TF_Tensor *tensor, std::string &format, char dim) {
  int idx = -1;
  if (format == "NCHW") {
    switch (dim) {
      case 'N': {
        idx = 0;
        break;
      }
      case 'C': {
        idx = 1;
        break;
      }
      case 'H': {
        idx = 2;
        break;
      }
      case 'W': {
        idx = 3;
        break;
      }
      default: {
        idx = -1;
      }
    }
  } else if (format == "NHWC") {
    switch (dim) {
      case 'N': {
        idx = 0;
        break;
      }
      case 'C': {
        idx = 3;
        break;
      }
      case 'H': {
        idx = 1;
        break;
      }
      case 'W': {
        idx = 2;
        break;
      }
      default: {
        idx = -1;
      }
    }
  } else {
    std::cerr << "Unsupport data_format now" << std::endl;
    return -1;
  }
  return TF_Dim(tensor, idx);
}

#define CHECK_CONSTRUCT_STATUS(ctx, status)         \
  do {                                              \
    if (TF_GetCode(status) != TF_OK) {              \
      TF_OpKernelConstruction_Failure(ctx, status); \
    }                                               \
  } while (0);

#define CHECK_CTX_STATUS(ctx, status)          \
  do {                                         \
    if (TF_GetCode(status) != TF_OK) {         \
      TF_OpKernelContext_Failure(ctx, status); \
    }                                          \
  } while (0);

namespace vulten_plugin {

static std::vector<uint32_t> spirv_im2col_valid;
static std::vector<uint32_t> spirv_im2col_same;
static std::vector<uint32_t> spirv_conv2d;
static std::vector<uint32_t> spirv_conv2dOld;

struct Im2colInfo {
  std::vector<int> dataVec;

  Im2colInfo() { dataVec.resize(18); }

  void printInfo() {
    std::cout << "Batches: " << getBatches() << "\n";
    std::cout << "Hight: " << getHight() << "\n";
    std::cout << "Width: " << getWidth() << "\n";
    std::cout << "Channels: " << getChannels() << "\n";

    std::cout << "Filter Hight: " << getfilterH() << "\n";
    std::cout << "Filter Width: " << getfilterW() << "\n";
    std::cout << "Filter In Channels: " << getfilterIn() << "\n";
    std::cout << "Filter Out Channels: " << getfilterOut() << "\n";

    std::cout << "Stride Hight: " << getstrideH() << "\n";
    std::cout << "Stride Width: " << getstrideW() << "\n";

    std::cout << "Dilation Hight: " << getdilationH() << "\n";
    std::cout << "Dilation Width: " << getdilationW() << "\n";

    std::cout << "Result Hight: " << getresHight() << "\n";
    std::cout << "Result Width: " << getresWidth() << "\n";
  }

  void setBatches(int batches) { dataVec[0] = batches; }
  int getBatches() { return dataVec[0]; }

  void setHight(int hight) { dataVec[1] = hight; }
  int getHight() { return dataVec[1]; }
  void setWidth(int width) { dataVec[2] = width; }
  int getWidth() { return dataVec[2]; }
  void setChannels(int channels) { dataVec[3] = channels; }
  int getChannels() { return dataVec[3]; }

  void setfilterH(int filterH) { dataVec[4] = filterH; }
  int getfilterH() { return dataVec[4]; }
  void setfilterW(int filterW) { dataVec[5] = filterW; }
  int getfilterW() { return dataVec[5]; }
  void setfilterIn(int filterIn) { dataVec[6] = filterIn; }
  int getfilterIn() { return dataVec[6]; }
  void setfilterOut(int filterOut) { dataVec[7] = filterOut; }
  int getfilterOut() { return dataVec[7]; }

  void setstrideH(int strideH) { dataVec[8] = strideH; }
  int getstrideH() { return dataVec[8]; }
  void setstrideW(int strideW) { dataVec[9] = strideW; }
  int getstrideW() { return dataVec[9]; }

  void setdilationH(int dilationH) { dataVec[10] = dilationH; }
  int getdilationH() { return dataVec[10]; }
  void setdilationW(int dilationW) { dataVec[11] = dilationW; }
  int getdilationW() { return dataVec[11]; }

  void setresHight(int resHight) { dataVec[12] = resHight; }
  int getresHight() { return dataVec[12]; }
  void setresWidth(int resWidth) { dataVec[13] = resWidth; }
  int getresWidth() { return dataVec[13]; }

  void setPadTop(int top) { dataVec[14] = top; };
  int getPadTop() { return dataVec[14]; };
  void setPadBottom(int bottom) { dataVec[15] = bottom; };
  int getPadBottom() { return dataVec[15]; };
  void setPadLeft(int left) { dataVec[16] = left; };
  int getPadLeft() { return dataVec[16]; };
  void setPadRight(int right) { dataVec[17] = right; };
  int getPadRight() { return dataVec[17]; };
};

template <class T>
struct Conv2dOp {
  Conv2dOp() : data_format_("") {}
  std::vector<int32_t> strides_;
  Padding padding_;
  std::string data_format_;
};

template <class T>
void *Conv2dOp_Create(TF_OpKernelConstruction *ctx) {
  auto kernel = new Conv2dOp<T>();

  StatusSafePtr status(TF_NewStatus());
  int32_t list_size = 0;
  int32_t total_size = 0;

  // Get strides
  TF_OpKernelConstruction_GetAttrSize(ctx, "strides", &list_size, &total_size,
                                      status.get());
  CHECK_CONSTRUCT_STATUS(ctx, status.get());
  kernel->strides_.resize(list_size);
  TF_OpKernelConstruction_GetAttrInt32List(
      ctx, "strides", kernel->strides_.data(), list_size, status.get());
  CHECK_CONSTRUCT_STATUS(ctx, status.get());

  // Get data_format
  TF_OpKernelConstruction_GetAttrSize(ctx, "data_format", &list_size,
                                      &total_size, status.get());
  CHECK_CONSTRUCT_STATUS(ctx, status.get());
  std::vector<char> format_vec(total_size);
  TF_OpKernelConstruction_GetAttrString(ctx, "data_format", format_vec.data(),
                                        total_size, status.get());
  CHECK_CONSTRUCT_STATUS(ctx, status.get());
  kernel->data_format_ = std::move(std::string(format_vec.data(), total_size));

  // Get padding
  TF_OpKernelConstruction_GetAttrSize(ctx, "padding", &list_size, &total_size,
                                      status.get());
  CHECK_CONSTRUCT_STATUS(ctx, status.get());
  std::vector<char> padding_vec(total_size);
  TF_OpKernelConstruction_GetAttrString(ctx, "padding", padding_vec.data(),
                                        total_size, status.get());
  CHECK_CONSTRUCT_STATUS(ctx, status.get());
  std::string padding_str(padding_vec.data(), total_size);
  if (padding_str == "VALID") {
    kernel->padding_ = Padding::VALID;
  } else if (padding_str == "SAME") {
    kernel->padding_ = Padding::SAME;
  } else {
    std::cerr << "Unsupported padding type: " << padding_str;
    return nullptr;
  }
  return kernel;
}

template <typename T>
void Conv2dOp_Delete(void *kernel) {
  if (kernel != nullptr) {
    delete static_cast<Conv2dOp<T> *>(kernel);
  }
}

template <typename T>
void Conv2dOp_Compute(void *kernel, TF_OpKernelContext *ctx) {
  Im2colInfo im2colInfo = Im2colInfo();

  StatusSafePtr status(TF_NewStatus());
  // Input tensor is of the following dimensions:
  // [ batch, in_rows, in_cols, in_depth ]
  TF_Tensor *input = nullptr;
  TF_GetInput(ctx, 0, &input, status.get());
  CHECK_CTX_STATUS(ctx, status.get());
  TensorSafePtr input_safe_ptr(input);
  auto in_ptr = static_cast<std::shared_ptr<kp::TensorT<float>> *>(
      TF_TensorData(input_safe_ptr.get()));

  im2colInfo.setBatches(TF_Dim(input, 0));
  im2colInfo.setHight(TF_Dim(input, 1));
  im2colInfo.setWidth(TF_Dim(input, 2));
  im2colInfo.setChannels(TF_Dim(input, 3));

  // Input filter is of the following dimensions:
  // [ filter_rows, filter_cols, in_depth, out_depth]
  TF_Tensor *filter = nullptr;
  TF_GetInput(ctx, 1, &filter, status.get());
  CHECK_CTX_STATUS(ctx, status.get());
  TensorSafePtr filter_safe_ptr(filter);
  auto filter_ptr = static_cast<std::shared_ptr<kp::TensorT<float>> *>(
      TF_TensorData(filter_safe_ptr.get()));

  im2colInfo.setfilterH(TF_Dim(filter, 0));
  im2colInfo.setfilterW(TF_Dim(filter, 1));
  im2colInfo.setfilterIn(TF_Dim(filter, 2));
  im2colInfo.setfilterOut(TF_Dim(filter, 3));

  if (TF_NumDims(input) != 4) {
    std::cerr << "input must be 4 dimensional" << std::endl;
    return;
  }
  if (TF_NumDims(filter) != 4) {
    std::cerr << "filter must be 4 dimensional" << std::endl;
    return;
  }

  for (int i = 0; i < 3; i++) {
    if (TF_Dim(filter, i) >= std::numeric_limits<int>::max()) {
      std::cerr << "filter too large" << std::endl;
      return;
    }
  }

  // The last dimension for input is in_depth. It must be the same as the
  // filter's in_depth.
  const int64_t in_depth = GetTensorDim(
      input, static_cast<Conv2dOp<T> *>(kernel)->data_format_, 'C');
  // if (in_depth != TF_Dim(filter, 3)) {
  //     std::cerr << "input and filter must have the same depth" << std::endl;
  //     return;
  // }

  // The last dimension for filter is out_depth.
  const int out_depth = static_cast<int>(TF_Dim(filter, 3));

  // The second dimension for input is rows/height.
  // The first dimension for filter is rows/height.
  const int64_t input_rows_raw = GetTensorDim(
      input, static_cast<Conv2dOp<T> *>(kernel)->data_format_, 'H');
  if (input_rows_raw >= std::numeric_limits<int>::max()) {
    std::cerr << "Input rows too large";
    return;
  }
  const int input_rows = static_cast<int>(input_rows_raw);
  const int filter_rows = static_cast<int>(TF_Dim(filter, 0));

  // The third dimension for input is columns/width.
  // The second dimension for filter is columns/width.
  const int64_t input_cols_raw = GetTensorDim(
      input, static_cast<Conv2dOp<T> *>(kernel)->data_format_, 'W');
  if (input_cols_raw >= std::numeric_limits<int>::max()) {
    std::cerr << "Input cols too large" << std::endl;
    return;
  }
  const int input_cols = static_cast<int>(input_cols_raw);
  const int filter_cols = static_cast<int>(TF_Dim(filter, 1));

  // The first dimension for input is batch.
  const int64_t batch_raw = GetTensorDim(
      input, static_cast<Conv2dOp<T> *>(kernel)->data_format_, 'N');
  if (batch_raw >= std::numeric_limits<int>::max()) {
    std::cerr << "batch is too large" << std::endl;
    return;
  }
  const int batch = static_cast<int>(batch_raw);

  // For now we take the stride from the second and third dimensions only (we
  // do not support striding on the batch or depth dimension).
  int stride_rows = 0;
  int stride_cols = 0;
  if (static_cast<Conv2dOp<T> *>(kernel)->data_format_ == "NCHW") {
    stride_rows = static_cast<Conv2dOp<T> *>(kernel)->strides_[2];
    stride_cols = static_cast<Conv2dOp<T> *>(kernel)->strides_[3];
  } else if (static_cast<Conv2dOp<T> *>(kernel)->data_format_ == "NHWC") {
    stride_rows = static_cast<Conv2dOp<T> *>(kernel)->strides_[1];
    stride_cols = static_cast<Conv2dOp<T> *>(kernel)->strides_[2];
  } else {
    std::cerr << "Unsupported data format" << std::endl;
    return;
  }
  im2colInfo.setstrideH(stride_rows);
  im2colInfo.setstrideW(stride_cols);

  int64_t out_rows = 0, out_cols = 0, pad_rows = 0, pad_cols = 0;
  if (!GetWindowedOutputSize(input_rows, filter_rows, 1, stride_rows,
                             static_cast<Conv2dOp<T> *>(kernel)->padding_,
                             &out_rows, &pad_rows)) {
    std::cerr << "Invalid filter size" << std::endl;
    return;
  }

  if (!GetWindowedOutputSize(input_cols, filter_cols, 1, stride_cols,
                             static_cast<Conv2dOp<T> *>(kernel)->padding_,
                             &out_cols, &pad_cols)) {
    std::cerr << "Invalid filter size" << std::endl;
    return;
  }
  auto output_size = batch * out_rows * out_cols * out_depth;
  std::vector<int64_t> out_shape;
  out_shape.push_back(batch);
  if (static_cast<Conv2dOp<T> *>(kernel)->data_format_ == "NCHW") {
    out_shape.push_back(out_depth);
    out_shape.push_back(out_rows);
    out_shape.push_back(out_cols);
  } else if (static_cast<Conv2dOp<T> *>(kernel)->data_format_ == "NHWC") {
    out_shape.push_back(out_rows);
    out_shape.push_back(out_cols);
    out_shape.push_back(out_depth);
  } else {
    std::cerr << "Unsupported data_foramt" << std::endl;
    return;
  }

  im2colInfo.setresHight(out_rows);
  im2colInfo.setresWidth(out_cols);

  if (static_cast<Conv2dOp<T> *>(kernel)->padding_ == Padding::VALID) {
    im2colInfo.setPadTop(0);
    im2colInfo.setPadBottom(0);
    im2colInfo.setPadLeft(0);
    im2colInfo.setPadRight(0);
  } else if (static_cast<Conv2dOp<T> *>(kernel)->padding_ = Padding::SAME) {
    if (im2colInfo.getHight() % im2colInfo.getstrideH() == 0) {
      int pad_along_height =
          std::max(im2colInfo.getfilterH() - im2colInfo.getstrideH(), 0);
      im2colInfo.setPadTop(std::floor(pad_along_height / 2.0F));
      im2colInfo.setPadBottom(std::ceil(pad_along_height / 2.0F));
    } else {
      int pad_along_height =
          std::max(im2colInfo.getfilterH() -
                       (im2colInfo.getHight() % im2colInfo.getstrideH()),
                   0);
      im2colInfo.setPadTop(std::floor(pad_along_height / 2.0F));
      im2colInfo.setPadBottom(std::ceil(pad_along_height / 2.0F));
    }
    if (im2colInfo.getWidth() % im2colInfo.getstrideW() == 0) {
      int pad_along_width =
          std::max(im2colInfo.getfilterW() - im2colInfo.getstrideW(), 0);
      im2colInfo.setPadLeft(std::floor(pad_along_width / 2.0F));
      im2colInfo.setPadRight(std::ceil(pad_along_width / 2.0F));
    } else {
      int pad_along_width =
          std::max(im2colInfo.getfilterW() -
                       (im2colInfo.getWidth() % im2colInfo.getstrideW()),
                   0);
      im2colInfo.setPadLeft(std::floor(pad_along_width / 2.0F));
      im2colInfo.setPadRight(std::ceil(pad_along_width / 2.0F));
    }
  }

  // Output tensor is of the following dimensions:
  // [ in_batch, out_rows, out_cols, out_depth ]``
  TensorSafePtr output_safe_ptr(TF_AllocateOutput(
      ctx, 0, TF_ExpectedOutputDataType(ctx, 0), out_shape.data(),
      out_shape.size(), sizeof(T) * output_size, status.get()));

  auto outputTensor = static_cast<std::shared_ptr<kp::TensorT<float>> *>(
      TF_TensorData(output_safe_ptr.get()));

  // If there is nothing to compute, return.
  if (output_size == 0) {
    return;
  }

  SP_Stream stream = TF_GetStream(ctx, status.get());
  // std::lock_guard<std::mutex> guard(stream->instance->testMutex);
  MutexScopeLock guard = MutexScopeLock(&stream->instance->mainQueueMutex);

  auto im2colInfoTen = stream->instance->mngr->tensorT<int>(
      {im2colInfo.dataVec}, kp::Tensor::TensorTypes::eDevice);
  stream->instance->mngr->sequence(stream->instance->mainQueue)
      ->record<kp::OpTensorSyncDevice>({im2colInfoTen})
      ->eval();

  int resChannels = 0;
  if (im2colInfo.getfilterIn() > 1) {
    resChannels = im2colInfo.getChannels();  // im2colInfo.getChannels() >=
                                             // im2colInfo.getfilterOut() ?
                                             // im2colInfo.getfilterOut() :
                                             // im2colInfo.getChannels();
  } else {
    resChannels = im2colInfo.getChannels() >= im2colInfo.getfilterOut()
                      ? im2colInfo.getfilterOut()
                      : im2colInfo.getChannels();
  }

  int filterArea = im2colInfo.getfilterH() * im2colInfo.getfilterW();
  int im2colRes = im2colInfo.getresHight() * im2colInfo.getresWidth() *
                  (filterArea * resChannels);
  im2colRes *= im2colInfo.getBatches();
  // std::cout << "im2col len: " << im2colRes << "\n";
  // std::cout << "Res Channeles " << resChannels << "\n";
  std::vector<float> stageVec(im2colRes);
  auto im2colTen = stream->instance->mngr->tensorT<float>(
      {stageVec}, kp::Tensor::TensorTypes::eDevice);
  // stream->instance->mngr->sequence()->record<kp::OpTensorSyncDevice>({im2colTen})->eval();

  std::shared_ptr<kp::Algorithm> im2colAlgo;
  if (static_cast<Conv2dOp<T> *>(kernel)->padding_ == Padding::VALID) {
    // std::cout << "padding: " << "VALID" << "\n";
    im2colAlgo = stream->instance->mngr->algorithm(
        {*in_ptr, im2colInfoTen, im2colTen}, spirv_im2col_valid,
        kp::Workgroup({uint32_t(im2colInfo.getBatches()),
                       uint32_t(im2colInfo.getresHight()),
                       uint32_t(im2colInfo.getresWidth())}));

  } else if (static_cast<Conv2dOp<T> *>(kernel)->padding_ = Padding::SAME) {
    // std::cout << "padding: " << "SAME" << "\n";
    im2colAlgo = stream->instance->mngr->algorithm(
        {*in_ptr, im2colInfoTen, im2colTen}, spirv_im2col_same,
        kp::Workgroup({uint32_t(im2colInfo.getBatches()),
                       uint32_t(im2colInfo.getresHight()),
                       uint32_t(im2colInfo.getresWidth())}));
  }

  stream->instance->mngr->sequence(stream->instance->mainQueue)
      ->record<kp::OpAlgoDispatch>(im2colAlgo)
      ->eval();

  // stream->instance->mngr->sequence()->record<kp::OpTensorSyncLocal>({im2colTen})->eval();
  // for(int i = 0; i < im2colTen->size(); i++){
  //     std::cout << im2colTen->vector()[i] << "|";

  // }
  // std::cout << "\n";
  // std::cout << "Filter IN: " << im2colInfo.getfilterIn() << "\n";
  // std::cout << "Pad top  : " << im2colInfo.getPadTop() << "\n";
  // std::cout << "Pad bot  : " << im2colInfo.getPadBottom() << "\n";
  // std::cout << "Pad left : " << im2colInfo.getPadLeft() << "\n";
  // std::cout << "Pad right: " << im2colInfo.getPadRight() << "\n";

  std::shared_ptr<kp::Algorithm> conv2dAlgo;
  if (im2colInfo.getfilterIn() == 1) {
    conv2dAlgo = stream->instance->mngr->algorithm(
        {im2colTen, *filter_ptr, im2colInfoTen, *outputTensor}, spirv_conv2dOld,
        kp::Workgroup(
            {uint32_t(im2colInfo.getBatches()),
             uint32_t((im2colRes / im2colInfo.getBatches()) / filterArea),
             uint32_t(std::ceil((float)im2colInfo.getfilterOut() /
                                im2colInfo.getChannels()))}));
  } else {
    conv2dAlgo = stream->instance->mngr->algorithm(
        {im2colTen, *filter_ptr, im2colInfoTen, *outputTensor}, spirv_conv2d,
        kp::Workgroup(
            {uint32_t(im2colInfo.getBatches()),
             uint32_t(im2colInfo.getresHight() * im2colInfo.getresWidth()),
             uint32_t(im2colInfo.getfilterOut())}));
  }
  // std::cout << "ASDASD: " << int((im2colRes / im2colInfo.getBatches()) /
  // filterArea) << "\n";

  stream->instance->mngr->sequence(stream->instance->mainQueue)
      ->record<kp::OpAlgoDispatch>(conv2dAlgo)
      ->eval();

  // im2colInfo.printInfo();
  // std::cout << "Using device: " << stream->deviceNum << "\n";
}

template <typename T>
void RegisterConvOpKernel(const char *device_type) {
  StatusSafePtr status(TF_NewStatus());
  auto *builder =
      TF_NewKernelBuilder("Conv2D", device_type, Conv2dOp_Create<T>,
                          &Conv2dOp_Compute<T>, &Conv2dOp_Delete<T>);
  // TF_DataType::TF_FLOAT

  TF_KernelBuilder_TypeConstraint(builder, "T", TF_DataType::TF_FLOAT,
                                  status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout << " Error while registering conv2d kernel with attribute T";
  TF_RegisterKernelBuilder("Conv2DOp", builder, status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout << " Error while registering conv2d kernel";
}

}  // namespace vulten_plugin

void RegisterDeviceConv2D(const char *device_type) {
  vulten_plugin::spirv_im2col_valid.resize(
      kp::shader_data::___shaders_im2colValid_comp_spv_len / 4);
  memcpy(&vulten_plugin::spirv_im2col_valid[0],
         kp::shader_data::___shaders_im2colValid_comp_spv,
         kp::shader_data::___shaders_im2colValid_comp_spv_len);

  vulten_plugin::spirv_im2col_same.resize(
      kp::shader_data::___shaders_im2colSame_comp_spv_len / 4);
  memcpy(&vulten_plugin::spirv_im2col_same[0],
         kp::shader_data::___shaders_im2colSame_comp_spv,
         kp::shader_data::___shaders_im2colSame_comp_spv_len);

  vulten_plugin::spirv_conv2d.resize(
      kp::shader_data::___shaders_conv2d_comp_spv_len / 4);
  memcpy(&vulten_plugin::spirv_conv2d[0],
         kp::shader_data::___shaders_conv2d_comp_spv,
         kp::shader_data::___shaders_conv2d_comp_spv_len);

  vulten_plugin::spirv_conv2dOld.resize(
      kp::shader_data::___shaders_conv2dOld_comp_spv_len / 4);
  memcpy(&vulten_plugin::spirv_conv2dOld[0],
         kp::shader_data::___shaders_conv2dOld_comp_spv,
         kp::shader_data::___shaders_conv2dOld_comp_spv_len);

  vulten_plugin::RegisterConvOpKernel<float>(device_type);
}