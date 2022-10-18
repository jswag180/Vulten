#pragma once

#include <chrono>

#include "gpuBackend.h"
#include "tensorflow/c/experimental/stream_executor/stream_executor.h"
#include "tensorflow/c/tf_status.h"

void SE_InitPluginFns(SE_PlatformRegistrationParams* const params,
                      TF_Status* const status);

struct SP_Stream_st {
  explicit SP_Stream_st(int devNum, gpuBackend* manager)
      : deviceNum(devNum), instance(manager) {}
  int deviceNum;
  gpuBackend* instance;
};

struct SP_Event_st {
  explicit SP_Event_st(void* event_h) : event_handle(event_h) {}
  void* event_handle;
};

struct SP_Timer_st {
  explicit SP_Timer_st(int id) : timer_handle(id) {}
  int timer_handle;
  std::chrono::_V2::system_clock::time_point start;
  std::chrono::_V2::system_clock::time_point stop;
};