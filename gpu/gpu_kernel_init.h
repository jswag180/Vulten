#pragma once

#include <iostream>

#include "Vulten.h"

void RegisterDeviceRelu(const char* device_type);
void RegisterDeviceSoftmax(const char* device_type);

void RegisterDeviceConv2D(const char* device_type);

void RegisterDeviceBiasAdd(const char* device_type);
void RegisterDeviceMatMul(const char* device_type);
void RegisterResourceApplyAdam(const char* device_type);
void RegisterDeviceAssignVariable(const char* device_type);
void RegisterAssignAddVariable(const char* device_type);
void RegisterDeviceReadVariableOp(const char* device_type);
void RegisterDeviceIdentityOp(const char* device_type);
void RegisterDeviceBasicOps(const char* device_type);
void RegisterDevicePow(const char* device_type);
void RegisterDeviceCast(const char* device_type);
void RegisterDeviceStridedSliceOp(const char* device_type);
void RegisterDeviceSparseSoftmaxCrossEntropyWithLogitsOp(
    const char* device_type);
void RegisterDeviceBiasAddGradOp(const char* device_type);
void RegisterDeviceKernels(const char* device_type);