#pragma once

const char* basic_source = R"(
#version 450

#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_scalar_block_layout : enable

#include "prelude.h"

layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer a { readonly  TYPE_0 x[]; };
layout(set = 0, binding = 1) buffer b { readonly  TYPE_0 y[]; };
layout(std430, set = 0, binding = 2) uniform c { readonly uint   strides[1000]; };
layout(std430, set = 0, binding = 3) uniform d { readonly uint   dims[500]; };
layout(set = 0, binding = 4) buffer e { writeonly TYPE_0 outData[]; };

#define OP_MUL        0
#define OP_ADD        1
#define OP_SUB        2
#define OP_DIV        3
#define OP_DIV_NO_NAN 4
#define OP_MAXIMUM    5
#define OP_MINIMUM    6
#define OP_DIV_REAL   7
layout(constant_id = 1) const uint op = 0;

layout(push_constant) uniform PushConstants {
    uint xRanks, yRanks, outRanks;
} push_const;

//x_strides, y_strides, out_strides
//dims will be 1 padded so they are the same shape
//x_dims, y_dims
void main(){
    
    uint xNumStrdies = push_const.xRanks + 1;
    uint yNumStrdies = push_const.yRanks + 1;
    uint outNumStrdies = push_const.outRanks + 1;
    uint ranks = outNumStrdies - 1;
    
    uint yStrideOffset = xNumStrdies;
    uint outStrideOffset = xNumStrdies + yNumStrdies;
    
    
    uint id = uint(gl_GlobalInvocationID.x);
    uint flatCord = id;
    
    uint outDim = 0;
    uint xIndex = 0;
    uint yIndex = 0;
    
    outDim = id / strides[outStrideOffset + 1];
    flatCord -= strides[outStrideOffset + 1] * outDim;
    if(dims[0] > 1)
		xIndex += outDim * strides[1];
	if(dims[push_const.xRanks] > 1)
		yIndex += outDim * strides[yStrideOffset + 1]; 
    for(uint i = 1; i < ranks - 1; i++){
        outDim = flatCord / strides[outStrideOffset + 1 + i];
        flatCord -= strides[outStrideOffset + 1 + i] * outDim;
        if(dims[i] > 1)
			xIndex += outDim * strides[i + 1];
		if(dims[i + push_const.xRanks] > 1)
			yIndex += outDim * strides[yStrideOffset + 1 + i];
    }
    if(ranks > 1){
        outDim = flatCord;
        if(dims[push_const.xRanks - 1] > 1)
			xIndex += outDim;
		if(dims[(push_const.xRanks + push_const.yRanks) - 1] > 1)
			yIndex += outDim;
    }

    TYPE_0 X = x[xIndex];    
    TYPE_0 Y = y[yIndex];
    
    if(op == OP_MUL){
        #if TYPE_NUM_0 == COMPLEX64
        outData[id] = cx_64_mul(X, Y);
        #elif TYPE_NUM_0 == COMPLEX128
        outData[id] = cx_128_mul(X, Y);
        #else
        outData[id] = X * Y;
        #endif
    }else if(op == OP_ADD){
        outData[id] = X + Y;
    }else if(op == OP_SUB){
        outData[id] = X - Y;
    }else if(op == OP_DIV){
        #if TYPE_NUM_0 == COMPLEX64
        outData[id] = cx_64_div(X, Y);
        #elif TYPE_NUM_0 == COMPLEX128
        outData[id] = cx_128_div(X, Y);
        #else
        outData[id] = X / Y;
        #endif
    }else if(op == OP_DIV_NO_NAN){
        if(Y == TYPE_0(0)){
            outData[id] = TYPE_0(0);
        }else{
            outData[id] = X / Y;
        }
    }else if(op == OP_MAXIMUM){
        outData[id] = max(X, Y);
    }else if(op == OP_MINIMUM){
        outData[id] = min(X, Y);
    }else if(op == OP_DIV_REAL){
        outData[id] = X / Y;
    }
}
)";