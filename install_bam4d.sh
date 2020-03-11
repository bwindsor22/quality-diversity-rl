#!/usr/bin/env bash
echo "creating environment"
python -m venv qd-bam4d-env
source qd-bam4d-env/bin/activate

echo "installing basic files"
pip install flatbuffers
pip install -r requirements.txt

echo "installing gym"
GYM_DIR=GVGAI_GYM_BAM
if [ -d "$GYM_DIR" ]; then rm -Rf $GYM_DIR; fi
mkdir $GYM_DIR
cd $GYM_DIR
git clone https://github.com/Bam4d/GVGAI_GYM.git
cd GVGAI_GYM/python/gvgai
pip install -e .
cd ../../../../
