#define FLOAT 0
#define INT 1
#define UINT 2
#define INT64_T 3
#define UINT64_T 4
#define INT8_T 5
#define UINT8_T 6
#define DOUBLE 7
#define HALF 8

#if TYPE_NUM_0 == INT64_T || TYPE_NUM_0 == UINT64_T || TYPE_NUM_1 == INT64_T || TYPE_NUM_1 == UINT64_T
    #extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
    #extension GL_EXT_shader_subgroup_extended_types_int64 : enable
#endif
#if TYPE_NUM_0 == INT8_T || TYPE_NUM_0 == UINT8_T || TYPE_NUM_1 == INT8_T || TYPE_NUM_1 == UINT8_T
    #extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable
    #extension GL_EXT_shader_subgroup_extended_types_int8 : enable
#endif
#if TYPE_NUM_0 == HALF || TYPE_NUM_1 == HALF
    #extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable
    #extension GL_EXT_shader_subgroup_extended_types_float16 : enable
#endif
