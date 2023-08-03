#pragma once

const char* transpose_source = R"(
#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : enable

#include "prelude.h"

//layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer a { readonly  TYPE_0 inData[]; };
layout(set = 0, binding = 1) buffer c { writeonly TYPE_0 outData[]; };

layout(push_constant) uniform PushConstants {
    uint hight, width;
} push_const;

void main(){
    uint i = gl_GlobalInvocationID.x / push_const.hight;
    uint j = gl_GlobalInvocationID.x % push_const.hight;
    outData[gl_GlobalInvocationID.x] = inData[push_const.width * j + i];
}
)";