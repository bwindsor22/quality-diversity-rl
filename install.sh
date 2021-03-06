#!/usr/bin/env bash
echo "creating ruben gvgai environment, should enable tests/test_ruben.py to pass"
python3 -m venv qd-env
source qd-env/bin/activate

echo "installing basic files"
pip install -r requirements.txt

echo "installing gym"
GYM_DIR=GVGAI_GYM
if [ -d "$GYM_DIR" ]; then rm -Rf $GYM_DIR; fi
git clone https://github.com/rubenrtorrado/GVGAI_GYM.git
cd GVGAI_GYM
pip install -e .
cd ..
