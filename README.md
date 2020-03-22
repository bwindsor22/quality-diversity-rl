# gameQD
This repository contains the code for the GIL project on general gameplay through quailty-diversity search



### Setup
A typical install and run might look like: 
* `git clone https://github.com/bwindsor22/quality-diversity-rl.git`
* `cd quality-diversity-rl`
* `bash install_bam4d.sh`
* ```export PYTHONPATH=`pwd`:$PYTHONPATH```
* `python evolution/run_mapelite_train.py --num_threads 9 --num_iter 50000`


#### testing setup
* Run `python -m unittest test/test_bam4d.py`. If you don't see commands and actions getting printed out, try these steps:
    * Check java: try `conda install -c conda-forge/label/gcc7 openjdk`. Check `java --version` shows java. Ensure `which java` points to the same java as `echo $JAVA_HOME`. 
    * Check gradle: `cd GVGAI_GYM_BAM/GVGAI_GYM/` and run `./gradlew --status`
    * If you've changed gradle or java, try reinstalling bam4d
