#pragma once

const char* softmax_source = R"(
#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : enable

#include "prelude.h"

layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer a { readonly  TYPE_0 inData[]; };
layout(set = 0, binding = 1) buffer b { readonly  TYPE_0 logitSums[]; };
layout(set = 0, binding = 2) buffer c { writeonly TYPE_0 outData[]; };

layout(push_constant) uniform PushConstants {
    uint numLogits;
} push_const;


void main(){
    outData[gl_GlobalInvocationID.x] = inData[gl_GlobalInvocationID.x] / logitSums[uint(floor(float(gl_GlobalInvocationID.x) / float(push_const.numLogits)))];
}
)";