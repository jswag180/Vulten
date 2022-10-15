#!/usr/bin/env bash

SOURCE_DIR=`dirname $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) `

python3 -m venv $SOURCE_DIR/venv
source $SOURCE_DIR/venv/bin/activate
pip3 install -r $SOURCE_DIR/requirements.txt
