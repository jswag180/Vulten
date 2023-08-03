#pragma once

const char* batchAdd_source = R"(
#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : enable

#include "prelude.h"

layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer a { readonly  TYPE_0 inData[]; };
layout(set = 0, binding = 1) buffer b { writeonly TYPE_0 outData[]; };

layout(push_constant) uniform PushConstants {
    uint numsInBatch;
} push_const;


void main(){
    TYPE_0 temp = TYPE_0(0);

    for(uint i = 0; i < push_const.numsInBatch; i++){
        temp += inData[gl_GlobalInvocationID.x * push_const.numsInBatch + i];
    }
    outData[gl_GlobalInvocationID.x] = temp;
}
)";