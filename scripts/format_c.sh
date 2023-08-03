#!/usr/bin/env bash

SOURCE_DIR=`dirname $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) `

clang-format --style=file -i $SOURCE_DIR/gpu/* $SOURCE_DIR/*.cpp $SOURCE_DIR/*.h
clang-format --style=file -i $SOURCE_DIR/Vulten_backend/*.cpp $SOURCE_DIR/Vulten_backend/*.h $SOURCE_DIR/Vulten_backend/ops/*.cpp $SOURCE_DIR/Vulten_backend/ops/*.h
clang-format --style=file -i $SOURCE_DIR/Vulten_backend/ops/relu/*
clang-format --style=file -i $SOURCE_DIR/Vulten_backend/ops/assign_add_sub/*
clang-format --style=file -i $SOURCE_DIR/Vulten_backend/ops/addn/*
clang-format --style=file -i $SOURCE_DIR/Vulten_backend/ops/basic/*
clang-format --style=file -i $SOURCE_DIR/Vulten_backend/ops/biasAdd/*
clang-format --style=file -i $SOURCE_DIR/Vulten_backend/ops/cast/*
clang-format --style=file -i $SOURCE_DIR/Vulten_backend/ops/exp/*
clang-format --style=file -i $SOURCE_DIR/Vulten_backend/ops/matMul/*
clang-format --style=file -i $SOURCE_DIR/Vulten_backend/ops/transpose/*
clang-format --style=file -i $SOURCE_DIR/Vulten_backend/ops/pow/*
clang-format --style=file -i $SOURCE_DIR/Vulten_backend/ops/reluGrad/*
clang-format --style=file -i $SOURCE_DIR/Vulten_backend/ops/resource_apply_adam/*