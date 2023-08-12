#pragma once

const char* multiFunc_source = R"(
#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : enable

#include "prelude.h"

layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer a { readonly  TYPE_0 inData[]; };
layout(set = 0, binding = 1) buffer b { writeonly TYPE_0 outData[]; };

#define OP_SQRT   0
#define OP_EXP    1
#define OP_LOG    2
#define OP_SQUARE 3
layout(push_constant) uniform PushConstants {
	uint op;
} push_const;

void main(){
    switch(push_const.op){
        case OP_SQRT:
            outData[gl_GlobalInvocationID.x] = sqrt(inData[gl_GlobalInvocationID.x]);
            break;
        case OP_EXP:
            outData[gl_GlobalInvocationID.x] = exp(inData[gl_GlobalInvocationID.x]);
            break;
        case OP_LOG:
            outData[gl_GlobalInvocationID.x] = log(inData[gl_GlobalInvocationID.x]);
            break;
        case OP_SQUARE:
            outData[gl_GlobalInvocationID.x] = inData[gl_GlobalInvocationID.x] * inData[gl_GlobalInvocationID.x];
            break;
    }
}
)";