cmake_minimum_required(VERSION 3.8)
include(FetchContent)

project( Vulten )

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(SOURCES "Vulten.cpp" "vulten_device.cpp" "vulten_kernel.cpp" "tensor_utills.cpp")

file(GLOB GPU_FIELS gpu/*)
list(APPEND SOURCES ${GPU_FIELS})

option(ENABLE_SCOPE_TIMERS "Enable timers for ops" ON)

set(VULTEN_LOG_LEVEL "2" CACHE STRING "Vulten log level")
add_subdirectory(Vulten_backend Vulten_backend/)

#Set the location of tensorflow site-packages wether it is is installed system/user or in a venv
#This will be used for linking, include
set(TF_LOCAL venv/lib/python3.10/site-packages)

add_library(Vulten SHARED ${SOURCES})

if(${ENABLE_SCOPE_TIMERS})
    message("-- Scope timers: enabled")
    add_compile_definitions(OP_TIMERS)
endif()

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

#Tensorflow
include_directories(${TF_LOCAL}/tensorflow/include)

target_link_directories(Vulten PUBLIC ${TF_LOCAL}/tensorflow/python)

find_library(PYWARP _pywrap_tensorflow_internal)

link_directories(${Vulkan_LIBRARIES})

target_link_libraries(Vulten Vulten_backend Vulkan::Vulkan :_pywrap_tensorflow_internal.so)


include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/gpu)
include_directories(${Vulkan_INCLUDE_DIRS})


SET(CMAKE_CXX_FLAGS "-pthread -fPIC" CACHE STRING "C++ flags" FORCE)

#Resulting .so should be placed into $ENV{HOME}/.local/lib/python3.10/site-packages/tensorflow-plugins
#or it should be packaged into a wheel
