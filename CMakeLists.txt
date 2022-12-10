cmake_minimum_required(VERSION 3.8)
include(FetchContent)

project( Vulten )

set(SOURCES "Vulten.cpp" "vulten_device.cpp" "vulten_kernel.cpp")

file(GLOB GPU_FIELS gpu/*)
list(APPEND SOURCES ${GPU_FIELS})

option(KOMPUTE_OPT_REPO_SUBMODULE_BUILD "Use the submodule repos instead of external package manager" 1)
option(KOMPUTE_OPT_ENABLE_SPDLOG "Extra compile flags for Kompute, see docs for full list" 0)
if(KOMPUTE_OPT_ENABLE_SPDLOG)
    set(KOMPUTE_EXTRA_CXX_FLAGS "${KOMPUTE_EXTRA_CXX_FLAGS} -DKOMPUTE_ENABLE_SPDLOG=1")
endif()
option(ENABLE_SCOPE_TIMERS "Enable timers for ops" OFF)

# It is necessary to pass the DEBUG or RELEASE flag accordingly to Kompute
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG=1 ${KOMPUTE_EXTRA_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DRELEASE=1 -DKOMPUTE_LOG_LEVEL=6 ${KOMPUTE_EXTRA_CXX_FLAGS}")

FetchContent_Declare(kompute
                     GIT_REPOSITORY https://github.com/KomputeProject/kompute.git
                     GIT_TAG        v0.8.1)
FetchContent_MakeAvailable(kompute)

#Set the location of tensorflow site-packages wether it is is installed system/user or in a venv
#This will be used for linking, include
set(TF_LOCAL venv/lib/python3.10/site-packages)

add_custom_target( cast
    COMMAND ${CMAKE_SOURCE_DIR}/venv/bin/python3 ${CMAKE_SOURCE_DIR}/scripts/genShaders.py -s ${CMAKE_SOURCE_DIR}/shaders/Cast.comp -o ${CMAKE_SOURCE_DIR}/shaders/headers/ -t float,int,uint,int64_t,uint64_t,int8_t,uint8_t,double
)
add_custom_target( addInPlace
    COMMAND ${CMAKE_SOURCE_DIR}/venv/bin/python3 ${CMAKE_SOURCE_DIR}/scripts/genShaders.py -s ${CMAKE_SOURCE_DIR}/shaders/AddInPlace.comp -o ${CMAKE_SOURCE_DIR}/shaders/headers/ -t float,int,uint,int64_t,uint64_t,int8_t,uint8_t,double
)
add_custom_target( basicOps
    COMMAND ${CMAKE_SOURCE_DIR}/venv/bin/python3 ${CMAKE_SOURCE_DIR}/scripts/genShaders.py -s ${CMAKE_SOURCE_DIR}/shaders/BasicOps.comp -o ${CMAKE_SOURCE_DIR}/shaders/headers/ -t float,int,uint,int64_t,uint64_t,int8_t,uint8_t,double
)
add_custom_target( biasAdd
    COMMAND ${CMAKE_SOURCE_DIR}/venv/bin/python3 ${CMAKE_SOURCE_DIR}/scripts/genShaders.py -s ${CMAKE_SOURCE_DIR}/shaders/BiasAdd.comp -o ${CMAKE_SOURCE_DIR}/shaders/headers/ -t float,int,uint,int64_t,uint64_t,int8_t,uint8_t,double
)
add_custom_target( conv2d
    COMMAND ${CMAKE_SOURCE_DIR}/venv/bin/python3 ${CMAKE_SOURCE_DIR}/scripts/genShaders.py -s ${CMAKE_SOURCE_DIR}/shaders/conv2d.comp -o ${CMAKE_SOURCE_DIR}/shaders/headers/ -t float,int,uint,int64_t,uint64_t,int8_t,uint8_t,double
    COMMAND ${CMAKE_SOURCE_DIR}/venv/bin/python3 ${CMAKE_SOURCE_DIR}/scripts/genShaders.py -s ${CMAKE_SOURCE_DIR}/shaders/conv2dOld.comp -o ${CMAKE_SOURCE_DIR}/shaders/headers/ -t float,int,uint,int64_t,uint64_t,int8_t,uint8_t,double
    COMMAND ${CMAKE_SOURCE_DIR}/venv/bin/python3 ${CMAKE_SOURCE_DIR}/scripts/genShaders.py -s ${CMAKE_SOURCE_DIR}/shaders/im2colSame.comp -o ${CMAKE_SOURCE_DIR}/shaders/headers/ -t float,int,uint,int64_t,uint64_t,int8_t,uint8_t,double
    COMMAND ${CMAKE_SOURCE_DIR}/venv/bin/python3 ${CMAKE_SOURCE_DIR}/scripts/genShaders.py -s ${CMAKE_SOURCE_DIR}/shaders/im2colValid.comp -o ${CMAKE_SOURCE_DIR}/shaders/headers/ -t float,int,uint,int64_t,uint64_t,int8_t,uint8_t,double
)
add_custom_target( transpose
    COMMAND ${CMAKE_SOURCE_DIR}/venv/bin/python3 ${CMAKE_SOURCE_DIR}/scripts/genShaders.py -s ${CMAKE_SOURCE_DIR}/shaders/Transpose.comp -o ${CMAKE_SOURCE_DIR}/shaders/headers/ -t float,int,uint,int64_t,uint64_t,int8_t,uint8_t,double
)
add_custom_target( matMul
    COMMAND ${CMAKE_SOURCE_DIR}/venv/bin/python3 ${CMAKE_SOURCE_DIR}/scripts/genShaders.py -s ${CMAKE_SOURCE_DIR}/shaders/MatMul.comp -o ${CMAKE_SOURCE_DIR}/shaders/headers/ -t float,int,uint,int64_t,uint64_t,int8_t,uint8_t,double
)
add_custom_target( pow
    COMMAND ${CMAKE_SOURCE_DIR}/venv/bin/python3 ${CMAKE_SOURCE_DIR}/scripts/genShaders.py -s ${CMAKE_SOURCE_DIR}/shaders/Pow.comp -o ${CMAKE_SOURCE_DIR}/shaders/headers/ -t float,int,uint,int64_t,uint64_t,int8_t,uint8_t,double
)
add_custom_target( relu
    COMMAND ${CMAKE_SOURCE_DIR}/venv/bin/python3 ${CMAKE_SOURCE_DIR}/scripts/genShaders.py -s ${CMAKE_SOURCE_DIR}/shaders/Relu.comp -o ${CMAKE_SOURCE_DIR}/shaders/headers/ -t float,int,uint,int64_t,uint64_t,int8_t,uint8_t,double
)
add_custom_target( applyAdam
    COMMAND ${CMAKE_SOURCE_DIR}/venv/bin/python3 ${CMAKE_SOURCE_DIR}/scripts/genShaders.py -s ${CMAKE_SOURCE_DIR}/shaders/ApplyAdam.comp -o ${CMAKE_SOURCE_DIR}/shaders/headers/ -t float,int,uint #int64,int8,double does not have pow...
)
add_custom_target(softmax
    COMMAND ${CMAKE_SOURCE_DIR}/venv/bin/python3 ${CMAKE_SOURCE_DIR}/scripts/genShaders.py -s ${CMAKE_SOURCE_DIR}/shaders/Softmax.comp -o ${CMAKE_SOURCE_DIR}/shaders/headers/ -t float
    COMMAND ${CMAKE_SOURCE_DIR}/venv/bin/python3 ${CMAKE_SOURCE_DIR}/scripts/genShaders.py -s ${CMAKE_SOURCE_DIR}/shaders/Exp.comp -o ${CMAKE_SOURCE_DIR}/shaders/headers/ -t float #int64,int8,double does not have exp...
    COMMAND ${CMAKE_SOURCE_DIR}/venv/bin/python3 ${CMAKE_SOURCE_DIR}/scripts/genShaders.py -s ${CMAKE_SOURCE_DIR}/shaders/BatchAdd.comp -o ${CMAKE_SOURCE_DIR}/shaders/headers/ -t float
)



add_library(Vulten SHARED ${SOURCES})


if(${ENABLE_SCOPE_TIMERS})
    message("-- Scope timers: enabled")
    add_compile_definitions(OP_TIMERS)
endif()

add_dependencies(Vulten cast addInPlace basicOps biasAdd conv2d transpose matMul pow relu applyAdam softmax)

find_package(Vulkan REQUIRED)

#Tensorflow
include_directories(${TF_LOCAL}/tensorflow/include)

target_link_directories(Vulten PUBLIC ${TF_LOCAL}/tensorflow/python)

find_library(PYWARP _pywrap_tensorflow_internal)

target_link_libraries(Vulten kompute::kompute Vulkan::Vulkan :_pywrap_tensorflow_internal.so)

if(KOMPUTE_OPT_ENABLE_SPDLOG)
    target_link_libraries(Vulten
        spdlog::spdlog)
endif()

include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/gpu)


SET(CMAKE_CXX_FLAGS "-pthread -fPIC -LTO" CACHE STRING "C++ flags" FORCE)

#Resulting .so should be placed into $ENV{HOME}/.local/lib/python3.10/site-packages/tensorflow-plugins
#or it should be packaged into a wheel
