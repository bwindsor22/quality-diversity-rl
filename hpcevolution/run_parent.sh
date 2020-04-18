#!/usr/bin/env bash
export PYTHONPATH=$(dirname `pwd`):$PYTHONPATH
echo "pythonpath $PYTHONPATH"
source ././../qd-env/bin/activate
python run_parent.py