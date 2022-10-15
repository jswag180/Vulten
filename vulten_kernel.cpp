#include <iostream>

#include "tensorflow/c/kernels.h"
//#include "cpu/cpu_kernel_init.h"
#include "Vulten.h"
#include "gpu_kernel_init.h"

void TF_InitKernel() { RegisterDeviceKernels(DEVICE_TYPE); }
// void TF_InitKernel() { }