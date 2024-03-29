#pragma once

const char* resource_apply_adam_source = R"(
#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : enable

#include "prelude.h"

layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0)  buffer a {          TYPE_0 var[]; };
layout(set = 0, binding = 1)  buffer b { readonly TYPE_0 m[]; };
layout(set = 0, binding = 2)  buffer c { readonly TYPE_0 v[]; };
layout(set = 0, binding = 3)  buffer e { readonly TYPE_0 beta1_power; };
layout(set = 0, binding = 4)  buffer f { readonly TYPE_0 beta2_power; };
layout(set = 0, binding = 5)  buffer g { readonly TYPE_0 lr; };
layout(set = 0, binding = 6)  buffer h { readonly TYPE_0 beta1; };
layout(set = 0, binding = 7)  buffer i { readonly TYPE_0 beta2; };
layout(set = 0, binding = 8)  buffer j { readonly TYPE_0 epsilon; };
layout(set = 0, binding = 9)  buffer k { readonly TYPE_0 grad[]; };

layout(constant_id = 1) const bool use_nesterov = false;

void main(){
    
    #if TYPE_NUM_0 == FLOAT || TYPE_NUM_0 == DOUBLE
        TYPE_0 lr_t = lr * (sqrt(1 - pow(beta2_power, max(1, gl_GlobalInvocationID.x))) / (1 - pow(beta1_power, max(1, gl_GlobalInvocationID.x))));
        TYPE_0 m_t  = beta1 * m[max(0, gl_GlobalInvocationID.x - 1)] + (1 - beta1) * grad[gl_GlobalInvocationID.x];
        TYPE_0 v_t  = beta2 * v[max(0, gl_GlobalInvocationID.x - 1)] + (1 - beta2) * pow(grad[gl_GlobalInvocationID.x], 2);
    #else
        TYPE_0 lr_t = TYPE_0(lr * (sqrt(1 - pow(beta2_power, max(1, gl_GlobalInvocationID.x))) / (1 - pow(beta1_power, max(1, gl_GlobalInvocationID.x)))));
        TYPE_0 m_t  = beta1 * m[max(0, gl_GlobalInvocationID.x - 1)] + TYPE_0(1 - beta1) * grad[gl_GlobalInvocationID.x];
        TYPE_0 v_t  = beta2 * v[max(0, gl_GlobalInvocationID.x - 1)] + TYPE_0(1 - beta2) * TYPE_0(pow(grad[gl_GlobalInvocationID.x], 2));
    #endif
    

    if(bool(use_nesterov)){
        #if TYPE_NUM_0 == FLOAT || TYPE_NUM_0 == DOUBLE
            var[gl_GlobalInvocationID.x] = var[gl_GlobalInvocationID.x] - (m_t * beta1 + grad[gl_GlobalInvocationID.x] * (1 - beta1)) * lr_t / (sqrt(v_t) + epsilon);
        #else
            var[gl_GlobalInvocationID.x] = var[gl_GlobalInvocationID.x] - (m_t * beta1 + grad[gl_GlobalInvocationID.x] * (1 - beta1)) * lr_t / TYPE_0(sqrt(v_t) + epsilon);
        #endif
    }else{
        #if TYPE_NUM_0 == FLOAT || TYPE_NUM_0 == DOUBLE
            var[gl_GlobalInvocationID.x] = var[gl_GlobalInvocationID.x] - m_t * lr_t / (sqrt(v_t) + epsilon);
        #else
            var[gl_GlobalInvocationID.x] = var[gl_GlobalInvocationID.x] - m_t * lr_t / TYPE_0(sqrt(v_t) + epsilon);
        #endif
    }
}
)";