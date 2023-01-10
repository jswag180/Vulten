#pragma once

#include <iostream>

#include "Vulten.h"

void RegisterDeviceRelu(const char* device_type);
void RegisterDeviceAssignVariable(const char* device_type);

void RegisterDeviceKernels(const char* device_type);