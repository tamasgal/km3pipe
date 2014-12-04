#!/bin/bash
echo "Setting environment for KM3Pipe"

ENV_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

if ! ( echo ${ENV_DIR} | egrep ${PYTHONPATH} > /dev/null ); then
    export PYTHONPATH=${ENV_DIR}:${PYTHONPATH}
fi

alias pipeinspector='python ${ENV_DIR}/pipeinspector/app.py'
