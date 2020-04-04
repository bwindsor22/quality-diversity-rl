#!/usr/bin/env bash
echo "creating environment"
python3 -m venv qd-env
source qd-env/bin/activate

echo "installing basic files"
pip3 install -r requirements.txt

echo "installing gym"
GYM_DIR=GVGAI_GYM
if [ -d "$GYM_DIR" ]; then rm -Rf $GYM_DIR; fi
git clone https://github.com/rubenrtorrado/GVGAI_GYM.git
cd GVGAI_GYM
pip3 install -e .
cd ..
