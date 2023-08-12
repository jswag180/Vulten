#pragma once

#include <iostream>

#include "Vulten.h"

void RegisterDeviceRelu(const char* device_type);
void RegisterDeviceAssignVariable(const char* device_type);
void RegisterResourceApplyAdam(const char* device_type);
void RegisterAssignAddSubVariable(const char* device_type);
void RegisterDeviceBasicOps(const char* device_type);
void RegisterDeviceCast(const char* device_type);
void RegisterDeviceReadVariableOp(const char* device_type);
void RegisterDeviceMatMul(const char* device_type);
void RegisterDeviceReluGrad(const char* device_type);
void RegisterDeviceBiasAddGradOp(const char* device_type);
void RegisterDeviceReduce(const char* device_type);
void RegisterDeviceAddn(const char* device_type);
void RegisterDeviceIdentity(const char* device_type);
void RegisterDevicePow(const char* device_type);
void RegisterDeviceIdentityN(const char* device_type);
void RegisterDeviceMultiFunc(const char* device_type);
void RegisterDeviceBiasAdd(const char* device_type);
void RegisterDeviceSoftmax(const char* device_type);

void RegisterDeviceKernels(const char* device_type);