#include <iostream>

#include "Vulten.h"
#include "gpu_kernel_init.h"
#include "tensorflow/c/kernels.h"

void TF_InitKernel() { RegisterDeviceKernels(DEVICE_TYPE); }