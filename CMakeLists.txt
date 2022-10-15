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

add_custom_target( generateShaderHeaders
    COMMAND python3 ${kompute_SOURCE_DIR}/scripts/convert_shaders.py -p ../shaders/ -s glslangValidator -c ${CMAKE_SOURCE_DIR}/shaders/headers/ -v
)



add_library(Vulten SHARED ${SOURCES})



add_dependencies(Vulten generateShaderHeaders)

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


#SET(CMAKE_CXX_FLAGS "-pthread -march=native -fPIC -LTO" CACHE STRING "POGGERS" FORCE)
SET(CMAKE_CXX_FLAGS "-pthread -fPIC -LTO")

#Resulting .so should be placed into $ENV{HOME}/.local/lib/python3.10/site-packages/tensorflow-plugins
#or it should be packaged into a wheel