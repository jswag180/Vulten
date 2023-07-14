cmake_minimum_required(VERSION 3.8)
include(FetchContent)

project( Vulten )

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(SOURCES "Vulten.cpp" "vulten_device.cpp" "vulten_kernel.cpp" "tensor_utills.cpp")

file(GLOB GPU_FIELS gpu/*)
list(APPEND SOURCES ${GPU_FIELS})

option(ENABLE_SCOPE_TIMERS "Enable timers for ops" ON)

set(VULTEN_LOG_LEVEL "2" CACHE STRING "Vulten log level")
set(VULTEN_DISABLE_16BIT false CACHE STRING "Disable float16/half/int16/uint16 ops")
set(VULTEN_DISABLE_8BIT false CACHE STRING "Disable int8/uint8/bool ops")
set(VULTEN_DISABLE_INT64 false CACHE STRING "Disable int64/uint64 ops")
set(VULTEN_DISABLE_DOUBLE false CACHE STRING "Disable float64/double and complex128 ops")
add_subdirectory(Vulten_backend Vulten_backend/)

#Set the location of tensorflow site-packages wether it is is installed system/user or in a venv
#This will be used for linking, include
set(TF_LOCAL venv/lib/python3.10/site-packages)

add_library(Vulten SHARED ${SOURCES})

if(${ENABLE_SCOPE_TIMERS})
    message("-- Scope timers: enabled")
    add_compile_definitions(OP_TIMERS)
endif()

find_package(Vulkan REQUIRED)

#Tensorflow
include_directories(${TF_LOCAL}/tensorflow/include)

target_link_directories(Vulten PUBLIC ${TF_LOCAL}/tensorflow/python)

find_library(PYWARP _pywrap_tensorflow_internal)

target_link_libraries(Vulten Vulten_backend Vulkan::Vulkan :_pywrap_tensorflow_internal.so)


include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/gpu)

if(VULTEN_DISABLE_16BIT)
    add_compile_definitions(VULTEN_DISABLE_16BIT)
endif()
if(VULTEN_DISABLE_8BIT)
    add_compile_definitions(VULTEN_DISABLE_8BIT)
endif()
if(VULTEN_DISABLE_INT64)
    add_compile_definitions(VULTEN_DISABLE_INT64)
endif()
if(VULTEN_DISABLE_DOUBLE)
    add_compile_definitions(VULTEN_DISABLE_DOUBLE)
endif()


SET(CMAKE_CXX_FLAGS "-pthread -fPIC" CACHE STRING "C++ flags" FORCE)

#Resulting .so should be placed into $ENV{HOME}/.local/lib/python3.10/site-packages/tensorflow-plugins
#or it should be packaged into a wheel
