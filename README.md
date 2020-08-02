# gameQD
This repository contains the code for the GIL project on general gameplay through quailty-diversity search



### Setup
A typical install and run might look like: 
* `git clone https://github.com/bwindsor22/quality-diversity-rl.git`
* `git checkout hpc`
* `cd quality-diversity-rl`
* `bash install_bam4d.sh`
* `cd hpcevolution`
* `bash run_parent.sh` # see "multithreading" below
* (In a new terminal)`bash start_child.sh 1`

### Multithreading and running on HPC
* To mimic the HPC when running locally, run different threads in different terminal windows. 
    *  For instance,
`bash run_parent.sh` starts the parent process, which runs the core map elites algorithm (Vanilla Map Elites, CMAME, etc)
    * `bash start_child.sh 1` starts a child with id 1 which can receive "work" (in the form of agents to be evaluated) and returns "Results" (fitness values) to the parent
    * `bash start_child.sh 2` starts a second child with id 2. We can start as many children as our cpu can handle.
    * Our parents and children communicate by writing work and results to disk, to a folder specified by the `--run_name` argument in the bash files. It is best to delete this file when restarting a run.
    * All parents and children have their own log files; see 'logs' folder for the output
* To run on HPC, replace the `.sh` commands with their `.sbatch` equivalents. `start_children.sh --agents=5` will start 5 sbatch processes.
    * Typically we begin with five agents, check everything is working then make a second call of`start_children.sh --agents=195` to scale to full load

#### Testing and debugging setup
* There are two gvgai libraries. `install.sh` installs the ruben library, which was tested mac. `install_bam4d.sh` installs the bam4d library, which was tested on unix. 
* **testing the bam4d gvgai library**: Run `python -m unittest test/test_bam4d.py`. If you don't see commands and actions getting printed out, try these steps:
    * Check java: try `conda install -c conda-forge/label/gcc7 openjdk` to get java. Check `java --version` shows java. Ensure `which java` points to the same java as `echo $JAVA_HOME`. `$JAVA_HOME/bin/java` should be the same as `which java`
    * Check gradle: `cd GVGAI_GYM_BAM/GVGAI_GYM/` and run `./gradlew --status`
    * If you've changed gradle or java, try reinstalling bam4d
    * Bam4d has only been tested on linux. If on Mac, run the Ruben library with install.sh
* Common errors
    * `module evolution not found`: make sure you set your python path, above
    * `module DQN has no attribute conv1`: ignore, this sometimes happens when starting up a lot of threads at once, but future threads will succeed

### Using lab hosts
* check ganglia (ask on slack for details) to watch your CPU usage. Ideally, you want to see `system` and `user` usage as high, but that `wait` is zero. If you see `wait`, you're overloading the cpus
* I recommend running from `tmux` and using one pane to run the script, while having `htop` open in another pane
