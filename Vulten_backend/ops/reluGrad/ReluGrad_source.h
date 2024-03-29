#pragma once

const char* reluGrad_source = R"(
#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : enable

#include "prelude.h"

layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer a { readonly  TYPE_0 gradients[];};
layout(set = 0, binding = 1) buffer b { readonly  TYPE_0 features[]; };
layout(set = 0, binding = 2) buffer c { writeonly TYPE_0 outData[];  };

void main() {
    if(features[gl_GlobalInvocationID.x] > TYPE_0(0)){
        outData[gl_GlobalInvocationID.x] = gradients[gl_GlobalInvocationID.x];
    }else{
        outData[gl_GlobalInvocationID.x] = TYPE_0(0);
    }
}
)";