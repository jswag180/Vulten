#pragma once

const char* xent_source = R"(
#version 450
#extension GL_GOOGLE_include_directive : enable

#include "prelude.h"

layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer a { readonly  TYPE_0 scratch[]; };
layout(set = 0, binding = 1) buffer b { readonly  TYPE_0 backprop[]; };
layout(set = 0, binding = 2) buffer c { readonly  TYPE_1 labels[]; };
layout(set = 0, binding = 3) buffer d { writeonly TYPE_0 outData[]; };


#define OP_LOSS 0
#define OP_GRAD 1
layout(push_constant) uniform PushConstants {
    uint numLogits;
    uint op;
} push_const;

void main() {
    uint id = uint(gl_GlobalInvocationID.x);
    uint x = id / push_const.numLogits;
    uint y = id % push_const.numLogits;

    if(push_const.op == OP_LOSS){
        if(labels[x] == TYPE_1(y)){
            outData[id] = log(scratch[x]) - backprop[id];
        }else{
            outData[id] = TYPE_0(0);
        }
    }else if(push_const.op == OP_GRAD){
        outData[id] = exp(backprop[id]) / scratch[x] - (TYPE_0(labels[x] == TYPE_1(y)));
    }
}
)";