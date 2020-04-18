#!/usr/bin/env bash
source ././../qd-env/bin/activate
export PYTHONPATH=$(dirname `pwd`):$PYTHONPATH
echo "pythonpath $PYTHONPATH"
python child.py --unique_id $1
