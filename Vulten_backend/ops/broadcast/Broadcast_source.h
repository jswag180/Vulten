#pragma once

const char* broadcast_source = R"(
#version 450
#extension GL_GOOGLE_include_directive : enable

#include "prelude.h"

layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer a { readonly  TYPE_0 inData[]; };
layout(set = 0, binding = 1) buffer b { readonly  uint   strides[]; };
layout(set = 0, binding = 2) buffer c { readonly  uint   dims[]; };
layout(set = 0, binding = 3) buffer d { writeonly TYPE_0 outData[]; };

layout(push_constant) uniform PushConstants {
    uint inRanks, outRanks;
} push_const;

void main() {
    uint xNumStrdies = push_const.inRanks + 1;
    uint outNumStrdies = push_const.outRanks + 1;
    uint outStrideOffset = xNumStrdies;
    uint ranks = outNumStrdies - 1;
    
    uint id = uint(gl_GlobalInvocationID.x);
    uint flatCord = id;
    uint outDim = 0;
    uint xIndex = 0;
    
    outDim = id / strides[outStrideOffset + 1];
    flatCord -= strides[outStrideOffset + 1] * outDim;
    if(dims[0] > 1)
		xIndex += outDim * strides[1];
    for(uint i = 1; i < ranks - 1; i++){
        outDim = flatCord / strides[outStrideOffset + 1 + i];
        flatCord -= strides[outStrideOffset + 1 + i] * outDim;
        if(dims[i] > 1)
			xIndex += outDim * strides[i + 1];
    }
    if(ranks > 1){
        outDim = flatCord;
        if(dims[push_const.inRanks - 1] > 1)
			xIndex += outDim;
    }

    outData[id] = inData[xIndex];
}
)";