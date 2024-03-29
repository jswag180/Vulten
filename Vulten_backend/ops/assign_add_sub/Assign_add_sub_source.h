#pragma once

const char* assign_add_sub_source = R"(
#version 450

#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : enable

#include "prelude.h"

layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer a { TYPE_0 tensor[]; };
layout(set = 0, binding = 1) buffer b { readonly TYPE_0 value[]; };

#define ADD 0
#define SUB 1
layout(push_constant) uniform PushConstants {
    int op;
} push_const;

void main(){
    if(push_const.op == ADD){
        tensor[gl_GlobalInvocationID.x] = tensor[gl_GlobalInvocationID.x] + value[gl_GlobalInvocationID.x];
    }else if(push_const.op == SUB){
        tensor[gl_GlobalInvocationID.x] = tensor[gl_GlobalInvocationID.x] - value[gl_GlobalInvocationID.x];
    }
}
)";