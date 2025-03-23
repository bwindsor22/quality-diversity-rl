mkdir hpcevolution/available_agents
mkdir hpcevolution/work_todo
mkdir hpcevolution/results
sbatch run_parent.sbatch
bash hpc_start_children.sh --agents=5
