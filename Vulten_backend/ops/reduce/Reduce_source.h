#pragma once

const char* reduce_source = R"(
#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : enable

#include "prelude.h"

layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer a {  TYPE_0 inData[];    };
layout(set = 0, binding = 1) buffer c {  TYPE_0 outData[];   };

layout(constant_id = 0) const uint localX = 0;

#define OP_SUM 0
#define OP_MAX 1
#define OP_MIN 2
layout(push_constant) uniform PushConstants {
    uint axi_size;
	uint adj_stride;
	uint adj_stride_adv; // adj_strides[axi_select + 1]
	uint op;
} push_const;

void main(){
	uint thread_id = uint(gl_GlobalInvocationID.x);
	uint indx = thread_id / push_const.adj_stride_adv * push_const.adj_stride + (thread_id % push_const.adj_stride_adv);
	
	outData[thread_id] = inData[indx];
	for(uint i = 1; i < push_const.axi_size; i++){
		if(push_const.op == OP_SUM){
			outData[thread_id] += inData[indx + (i * push_const.adj_stride_adv)];
		}else if(push_const.op == OP_MAX){
			outData[thread_id] = max(outData[thread_id], inData[indx + (i * push_const.adj_stride_adv)]);
		}else if(push_const.op == OP_MIN){
			outData[thread_id] = min(outData[thread_id], inData[indx + (i * push_const.adj_stride_adv)]);
		}
	}
}
)";