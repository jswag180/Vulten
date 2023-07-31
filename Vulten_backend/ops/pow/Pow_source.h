#pragma once

const char* pow_source = R"(
#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : enable

#include "prelude.h"

layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer a { readonly  TYPE_0 x[]; };
layout(set = 0, binding = 1) buffer b { readonly  TYPE_0 y[]; };
layout(set = 0, binding = 2) buffer c { writeonly TYPE_0 outData[]; };

layout(push_constant) uniform PushConstants {
    uint scalar;
} push_const;

void power(in TYPE_0 val, in TYPE_0 ex, inout TYPE_0 result){
    result = val;

    if(ex == TYPE_0(0)){
        result = TYPE_0(1);
        return;
    }else if (val == TYPE_0(0)){
        result = TYPE_0(0);
        return;
    }

    for(uint i = 0; i < ex - 1; i++){
        result *= val;
    }
}

void main(){
    if(push_const.scalar == 1){
        power(x[0], y[gl_GlobalInvocationID.x], outData[gl_GlobalInvocationID.x]);
    }else if(push_const.scalar == 2){
        power(x[gl_GlobalInvocationID.x], y[0], outData[gl_GlobalInvocationID.x]);
    }else{
        power(x[gl_GlobalInvocationID.x], y[gl_GlobalInvocationID.x], outData[gl_GlobalInvocationID.x]);
    }
}
)";