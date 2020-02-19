pip install -r requirements.txt

GYM_DIR=GVGAI_GYM
if [ -d "$GYM_DIR" ]; then rm -Rf $GYM_DIR; fi
git clone https://github.com/rubenrtorrado/GVGAI_GYM.git
cd GVGAI_GYM
pip install -e .
cd ..
