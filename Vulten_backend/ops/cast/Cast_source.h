#pragma once

const char* cast_source = R"(
#version 450

#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : enable

#include "prelude.h"

//layout(local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer a { readonly TYPE_0 inData[]; };
layout(set = 0, binding = 1) buffer b { writeonly TYPE_1 outData[]; };

void main(){
    #if TYPE_NUM_1 == COMPLEX64 || TYPE_NUM_1 == COMPLEX128
    outData[gl_GlobalInvocationID.x] = TYPE_1(inData[gl_GlobalInvocationID.x], 0.0);
    #elif TYPE_NUM_1 == BOOL
    #if TYPE_NUM_0 == COMPLEX128 || TYPE_NUM_0 == COMPLEX64 
    if(inData[gl_GlobalInvocationID.x].x == TYPE_0(0).x){
        outData[gl_GlobalInvocationID.x] = TYPE_1(0);
        return;
    }
    #else
    if(inData[gl_GlobalInvocationID.x] == TYPE_0(0)){
        outData[gl_GlobalInvocationID.x] = TYPE_1(0);
        return;
    }
    #endif
        outData[gl_GlobalInvocationID.x] = TYPE_1(1);
    #else
    outData[gl_GlobalInvocationID.x] = TYPE_1(inData[gl_GlobalInvocationID.x]);
    #endif
}
)";