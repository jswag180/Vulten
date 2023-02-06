#include "Vulten.h"

#include "tensorflow/c/experimental/stream_executor/stream_executor.h"
#include "vulten_device.h"

void SE_InitPlugin(SE_PlatformRegistrationParams* const params,
                   TF_Status* const status) {
  params->platform->struct_size = SP_PLATFORM_STRUCT_SIZE;
  params->platform->name = DEVICE_NAME;
  params->platform->type = DEVICE_TYPE;

  SE_InitPluginFns(params, status);
}