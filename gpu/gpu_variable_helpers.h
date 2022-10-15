#pragma once

#include <iostream>
#include <memory>

#include "gpuBackend.h"
#include "tensorflow/c/kernels.h"
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

namespace varHelpers {

static void copyFunc(TF_OpKernelContext *ctx, TF_Tensor *source,
                     TF_Tensor *dest) {
  StatusSafePtr status(TF_NewStatus());
  SP_Stream stream = TF_GetStream(ctx, status.get());
  // std::lock_guard<std::mutex> guard(stream->instance->testMutex);
  MutexScopeLock guard = MutexScopeLock(&stream->instance->mainQueueMutex);

  TensorSafePtr source_safe_ptr(source);
  auto source_ptr = static_cast<std::shared_ptr<kp::TensorT<float>> *>(
      TF_TensorData(source_safe_ptr.get()));
  // std::cout << "source: " << source_ptr << "\n";

  TensorSafePtr dest_safe_ptr(dest);
  auto dest_ptr = static_cast<std::shared_ptr<kp::TensorT<float>> *>(
      TF_TensorData(dest_safe_ptr.get()));
  // std::cout << "dest: " << dest_ptr << "\n";

  stream->instance->mngr->sequence(stream->instance->mainQueue)
      ->record<kp::OpTensorCopy>({*source_ptr, *dest_ptr})
      ->eval();

  // std::cout << "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC\n";
  // exit(-1);
}

}  // namespace varHelpers
