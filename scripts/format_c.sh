#!/usr/bin/env bash

SOURCE_DIR=`dirname $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) `

clang-format --style=file -i $SOURCE_DIR/gpu/* $SOURCE_DIR/*.cpp $SOURCE_DIR/*.h