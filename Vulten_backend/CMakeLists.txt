cmake_minimum_required(VERSION 3.8)

project( Vulten_backend )

set(CMAKE_CXX_STANDARD 17)

# 0:None 1:Info 2:Debug
set(VULTEN_LOG_LEVEL "0" CACHE STRING "Vulten log level")

file(GLOB SOURCES ${PROJECT_SOURCE_DIR}/*.cpp)
file(GLOB OPS ${PROJECT_SOURCE_DIR}/ops/*.cpp)

add_custom_target( prelude
    COMMAND ${CMAKE_SOURCE_DIR}/venv/bin/python3 ${CMAKE_SOURCE_DIR}/scripts/to_header.py -s ${CMAKE_SOURCE_DIR}/shaders/prelude.h -o ${CMAKE_SOURCE_DIR}/shaders/headers/
)
add_custom_target( cast
    COMMAND ${CMAKE_SOURCE_DIR}/venv/bin/python3 ${CMAKE_SOURCE_DIR}/scripts/to_header.py -s ${CMAKE_SOURCE_DIR}/shaders/Cast.comp -o ${CMAKE_SOURCE_DIR}/shaders/headers/
)
add_custom_target( addSubInPlace
    COMMAND ${CMAKE_SOURCE_DIR}/venv/bin/python3 ${CMAKE_SOURCE_DIR}/scripts/to_header.py -s ${CMAKE_SOURCE_DIR}/shaders/AddSubInPlace.comp -o ${CMAKE_SOURCE_DIR}/shaders/headers/  
)
add_custom_target( basicOps
    COMMAND ${CMAKE_SOURCE_DIR}/venv/bin/python3 ${CMAKE_SOURCE_DIR}/scripts/to_header.py -s ${CMAKE_SOURCE_DIR}/shaders/BasicOps.comp -o ${CMAKE_SOURCE_DIR}/shaders/headers/  
)
add_custom_target( biasAdd
    COMMAND ${CMAKE_SOURCE_DIR}/venv/bin/python3 ${CMAKE_SOURCE_DIR}/scripts/to_header.py -s ${CMAKE_SOURCE_DIR}/shaders/BiasAdd.comp -o ${CMAKE_SOURCE_DIR}/shaders/headers/  
)
add_custom_target( conv2d
    COMMAND ${CMAKE_SOURCE_DIR}/venv/bin/python3 ${CMAKE_SOURCE_DIR}/scripts/to_header.py -s ${CMAKE_SOURCE_DIR}/shaders/conv2d.comp -o ${CMAKE_SOURCE_DIR}/shaders/headers/  
    COMMAND ${CMAKE_SOURCE_DIR}/venv/bin/python3 ${CMAKE_SOURCE_DIR}/scripts/to_header.py -s ${CMAKE_SOURCE_DIR}/shaders/conv2dOld.comp -o ${CMAKE_SOURCE_DIR}/shaders/headers/  
    COMMAND ${CMAKE_SOURCE_DIR}/venv/bin/python3 ${CMAKE_SOURCE_DIR}/scripts/to_header.py -s ${CMAKE_SOURCE_DIR}/shaders/im2colSame.comp -o ${CMAKE_SOURCE_DIR}/shaders/headers/  
    COMMAND ${CMAKE_SOURCE_DIR}/venv/bin/python3 ${CMAKE_SOURCE_DIR}/scripts/to_header.py -s ${CMAKE_SOURCE_DIR}/shaders/im2colValid.comp -o ${CMAKE_SOURCE_DIR}/shaders/headers/  
)
add_custom_target( transpose
    COMMAND ${CMAKE_SOURCE_DIR}/venv/bin/python3 ${CMAKE_SOURCE_DIR}/scripts/to_header.py -s ${CMAKE_SOURCE_DIR}/shaders/Transpose.comp -o ${CMAKE_SOURCE_DIR}/shaders/headers/  
)
add_custom_target( matMul
    COMMAND ${CMAKE_SOURCE_DIR}/venv/bin/python3 ${CMAKE_SOURCE_DIR}/scripts/to_header.py -s ${CMAKE_SOURCE_DIR}/shaders/MatMul.comp -o ${CMAKE_SOURCE_DIR}/shaders/headers/  
)
add_custom_target( pow
    COMMAND ${CMAKE_SOURCE_DIR}/venv/bin/python3 ${CMAKE_SOURCE_DIR}/scripts/to_header.py -s ${CMAKE_SOURCE_DIR}/shaders/Pow.comp -o ${CMAKE_SOURCE_DIR}/shaders/headers/  
)
add_custom_target( relu
    COMMAND ${CMAKE_SOURCE_DIR}/venv/bin/python3 ${CMAKE_SOURCE_DIR}/scripts/to_header.py -s ${CMAKE_SOURCE_DIR}/shaders/Relu.comp -o ${CMAKE_SOURCE_DIR}/shaders/headers/
    COMMAND ${CMAKE_SOURCE_DIR}/venv/bin/python3 ${CMAKE_SOURCE_DIR}/scripts/to_header.py -s ${CMAKE_SOURCE_DIR}/shaders/ReluGrad.comp -o ${CMAKE_SOURCE_DIR}/shaders/headers/  
)
add_custom_target( applyAdam
    COMMAND ${CMAKE_SOURCE_DIR}/venv/bin/python3 ${CMAKE_SOURCE_DIR}/scripts/to_header.py -s ${CMAKE_SOURCE_DIR}/shaders/ApplyAdam.comp -o ${CMAKE_SOURCE_DIR}/shaders/headers/  #float,int,uint #extended types do not have pow...
)
add_custom_target(softmax
    COMMAND ${CMAKE_SOURCE_DIR}/venv/bin/python3 ${CMAKE_SOURCE_DIR}/scripts/to_header.py -s ${CMAKE_SOURCE_DIR}/shaders/Softmax.comp -o ${CMAKE_SOURCE_DIR}/shaders/headers/  #float
    COMMAND ${CMAKE_SOURCE_DIR}/venv/bin/python3 ${CMAKE_SOURCE_DIR}/scripts/to_header.py -s ${CMAKE_SOURCE_DIR}/shaders/Exp.comp -o ${CMAKE_SOURCE_DIR}/shaders/headers/  #float #int64,int8,double does not have exp...
    COMMAND ${CMAKE_SOURCE_DIR}/venv/bin/python3 ${CMAKE_SOURCE_DIR}/scripts/to_header.py -s ${CMAKE_SOURCE_DIR}/shaders/BatchAdd.comp -o ${CMAKE_SOURCE_DIR}/shaders/headers/  #float
)
add_custom_target(biasAddGrad
    COMMAND ${CMAKE_SOURCE_DIR}/venv/bin/python3 ${CMAKE_SOURCE_DIR}/scripts/to_header.py -s ${CMAKE_SOURCE_DIR}/shaders/BiasAddGrad.comp -o ${CMAKE_SOURCE_DIR}/shaders/headers/
)
add_custom_target(sum
    COMMAND ${CMAKE_SOURCE_DIR}/venv/bin/python3 ${CMAKE_SOURCE_DIR}/scripts/to_header.py -s ${CMAKE_SOURCE_DIR}/shaders/Sum.comp -o ${CMAKE_SOURCE_DIR}/shaders/headers/
)

add_library(Vulten_backend STATIC ${SOURCES} ${OPS})

add_dependencies(Vulten_backend prelude cast addSubInPlace basicOps biasAdd conv2d transpose matMul pow relu applyAdam softmax biasAddGrad sum)

if(VULTEN_LOG_LEVEL EQUAL "0")
    message("-- Vulten log level: 0")
    add_compile_definitions(VULTEN_LOG_LEVEL=0)
elseif(VULTEN_LOG_LEVEL EQUAL "1")
    message("-- Vulten log level: 1")
    add_compile_definitions(VULTEN_LOG_LEVEL=1)
elseif(VULTEN_LOG_LEVEL EQUAL "2")
    message("-- Vulten log level: 2")
    add_compile_definitions(VULTEN_LOG_LEVEL=2)
endif()

find_package(Vulkan REQUIRED)

target_link_libraries(Vulten_backend Vulkan::Vulkan shaderc_shared)

include_directories(${CMAKE_SOURCE_DIR})

SET(CMAKE_CXX_FLAGS "-pthread -fPIC" CACHE STRING "C++ flags" FORCE)
