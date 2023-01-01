cmake_minimum_required(VERSION 3.8)

project( Vulten_backend )

set(CMAKE_CXX_STANDARD 11)

# 0:None 1:Info 2:Debug
set(VULTEN_LOG_LEVEL "0" CACHE STRING "Vulten log level")

file(GLOB SOURCES ${PROJECT_SOURCE_DIR}/*.cpp)
file(GLOB OPS ${PROJECT_SOURCE_DIR}/ops/*.cpp)

add_library(Vulten_backend STATIC ${SOURCES} ${OPS})

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

target_link_libraries(Vulten_backend Vulkan::Vulkan)

include_directories(${CMAKE_SOURCE_DIR})

SET(CMAKE_CXX_FLAGS "-pthread -fPIC" CACHE STRING "C++ flags" FORCE)
