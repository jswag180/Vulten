#pragma once

const char* matMul_source = R"(
#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : enable

#include "prelude.h"

//layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer a { readonly  TYPE_0 aData[]; };
layout(set = 0, binding = 1) buffer b { readonly  TYPE_0 bData[]; };
layout(set = 0, binding = 2) buffer c { writeonly TYPE_0 outData[]; };

layout(push_constant) uniform PushConstants {
    uint aY, bY;
} push_const;

uint unFlatToFlat(in uint x, in uint y, in uint width){
    return x * width + y;
}

void main() {
    TYPE_0 dotProd = TYPE_0(0);

    for(uint i = 0; i < push_const.aY; i++){
        dotProd += aData[unFlatToFlat(gl_GlobalInvocationID.x, i, push_const.aY)] * bData[unFlatToFlat(i, gl_GlobalInvocationID.y, push_const.bY)];
    }

    outData[gl_GlobalInvocationID.x * gl_NumWorkGroups.y + gl_GlobalInvocationID.y] = dotProd;
}
)";