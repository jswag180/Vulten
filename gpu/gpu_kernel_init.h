#pragma once

#include <iostream>

#include "Vulten.h"

void RegisterDeviceRelu(const char* device_type);
void RegisterDeviceAssignVariable(const char* device_type);
void RegisterResourceApplyAdam(const char* device_type);
void RegisterAssignAddSubVariable(const char* device_type);
void RegisterDeviceBasicOps(const char* device_type);
void RegisterDeviceCast(const char* device_type);

void RegisterDeviceKernels(const char* device_type);
