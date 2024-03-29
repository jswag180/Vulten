#pragma once

const char* relu_source = R"(
#version 450
#extension GL_GOOGLE_include_directive : enable

#include "prelude.h"

layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer a { readonly  TYPE_0 inData[]; };
layout(set = 0, binding = 1) buffer b { writeonly TYPE_0 outData[]; };

void main() {
    if(inData[gl_GlobalInvocationID.x] > TYPE_0(0)){
        outData[gl_GlobalInvocationID.x] = inData[gl_GlobalInvocationID.x];
    }else{
        outData[gl_GlobalInvocationID.x] = TYPE_0(0);
    }
}
)";