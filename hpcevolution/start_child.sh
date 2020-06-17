#!/usr/bin/env bash
#source ././../qd-env/bin/activate
export PYTHONPATH=$(dirname `pwd`):$PYTHONPATH
echo "pythonpath $PYTHONPATH"
python child.py --unique_id $1 --gvgai_version GVGAI_BAM4D --game gvgai-zelda --run_name trial_run
