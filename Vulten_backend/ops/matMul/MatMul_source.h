#pragma once

const char* matMul_source = R"(
#version 450
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_control_flow_attributes : enable

#include "prelude.h"

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer a { readonly  TYPE_0 aData[]; };
layout(set = 0, binding = 1) buffer b { readonly  TYPE_0 bData[]; };
layout(set = 0, binding = 2) buffer c {  TYPE_0 outData[]; };// writeonly

layout(constant_id = 0) const uint localX = 0;
layout(constant_id = 1) const uint localY = 0;
layout(constant_id = 2) const uint blockSizeX = 4;
layout(constant_id = 3) const uint blockSizeY = 4;
layout(constant_id = 4) const uint bkCont = 1;
layout(constant_id = 5) const bool transA = false;
layout(constant_id = 6) const bool transB = false;

layout(push_constant) uniform PushConstants {
    uvec2 aDims;
    uvec2 bDims;
};

uint unFlatToFlat(in uint x, in uint y, in uint width){
    return x * width + y;
}

uint unFlatToFlatTrans(in uint x, in uint y, in uint width, in uint hight){
    uint i = (x * width + y) / width;//hight
    uint j = (x * width + y) % width;//hight
    return hight * j + i;//width
    
}

void main() {
    uint bi = uint(gl_GlobalInvocationID.x);
    uint bj = uint(gl_GlobalInvocationID.y);

    
    #ifdef UNROLL_BK
    [[unroll]]
    #endif
    for(uint bk = 0; bk < bkCont; bk++){
        [[unroll]] for(uint i = 0; i < blockSizeX; i++){
            [[unroll]] for(uint j = 0; j < blockSizeY; j++){
                uint cIndx = unFlatToFlat(bi * blockSizeX + i, bj * blockSizeY + j, bDims.y);
                TYPE_0 partial = TYPE_0(0);
                [[unroll]] for(uint k = 0; k < blockSizeX; k++){
                    uint aIndx = 0;
                    if(transA){
                        aIndx = unFlatToFlatTrans(bi * blockSizeX + i, bk * blockSizeX + k, aDims.y, aDims.x);  
                    }else{
                        aIndx = unFlatToFlat(bi * blockSizeX + i, bk * blockSizeX + k, aDims.y);  
                    }
                    uint bIndx = 0;
                    if(transB){
                        bIndx = unFlatToFlatTrans(bk * blockSizeX + k, bj * blockSizeY + j, bDims.y, bDims.x);
                    }else{
                        bIndx = unFlatToFlat(bk * blockSizeX + k, bj * blockSizeY + j, bDims.y);
                    }
                    #if TYPE_NUM_0 == COMPLEX64
                    partial += cx_64_mul(aData[aIndx], bData[bIndx]);
                    #elif TYPE_NUM_0 == COMPLEX128
                    partial += cx_128_mul(aData[aIndx], bData[bIndx]);
                    #else
                    partial += aData[aIndx] * bData[bIndx];
                    #endif
                }
                outData[cIndx] += partial;
            }
        }
    }
}
)";