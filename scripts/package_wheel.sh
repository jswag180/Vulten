#!/usr/bin/env bash

SOURCE_DIR=`dirname $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) `

FILE=$SOURCE_DIR/build/libVulten.so
if test -f "$FILE"; then
    cp $FILE $SOURCE_DIR/package/tensorflow-plugins/libVulten.so
else
    echo "Error $FILE does not exists"
    exit
fi

rm $SOURCE_DIR/package/dist/*

source $SOURCE_DIR/venv/bin/activate
(cd $SOURCE_DIR/package/ && python -m poetry build)

pip install --upgrade --no-deps --force-reinstall $SOURCE_DIR/package/dist/*.whl