#define FLOAT 0
#define INT 3
#define UINT 22
#define INT64_T 9
#define UINT64_T 23
#define INT8_T 6
#define UINT8_T 4
#define DOUBLE 2
#define HALF 19
#define INT16_T 5
#define UINT16_T 17
#define COMPLEX64 8
#define COMPLEX128 18
#define BOOL 10

#define bool8 uint8_t

#if TYPE_NUM_0 == INT64_T || TYPE_NUM_0 == UINT64_T || TYPE_NUM_1 == INT64_T || TYPE_NUM_1 == UINT64_T
    #extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
    #extension GL_EXT_shader_subgroup_extended_types_int64 : enable
#endif
#if TYPE_NUM_0 == INT16_T || TYPE_NUM_0 == UINT16_T || TYPE_NUM_1 == INT16_T || TYPE_NUM_1 == UINT16_T
    #extension GL_EXT_shader_explicit_arithmetic_types_int16 : enable
    #extension GL_EXT_shader_subgroup_extended_types_int16 : enable
#endif
#if TYPE_NUM_0 == INT8_T || TYPE_NUM_0 == UINT8_T || TYPE_NUM_1 == INT8_T || TYPE_NUM_1 == UINT8_T || TYPE_NUM_0 == BOOL || TYPE_NUM_1 == BOOL
    #extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable
    #extension GL_EXT_shader_subgroup_extended_types_int8 : enable
#endif
#if TYPE_NUM_0 == HALF || TYPE_NUM_1 == HALF
    #extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable
    #extension GL_EXT_shader_subgroup_extended_types_float16 : enable
#endif
//Complex implemenrations adopted from https://github.com/julesb/glsl-util
#if TYPE_NUM_0 == COMPLEX64 || TYPE_NUM_1 == COMPLEX64
    #define cx_64 vec2

    #define cx_64_mul(a, b) cx_64(a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x)
    #define cx_64_div(a, b) cx_64(((a.x*b.x+a.y*b.y)/(b.x*b.x+b.y*b.y)),((a.y*b.x-a.x*b.y)/(b.x*b.x+b.y*b.y)))
#endif
#if TYPE_NUM_0 == COMPLEX128 || TYPE_NUM_1 == COMPLEX128
    #define cx_128 dvec2

    #define cx_128_mul(a, b) cx_128(a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x)
    #define cx_128_div(a, b) cx_128(((a.x*b.x+a.y*b.y)/(b.x*b.x+b.y*b.y)),((a.y*b.x-a.x*b.y)/(b.x*b.x+b.y*b.y)))
#endif