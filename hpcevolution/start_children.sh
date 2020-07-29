#!/usr/bin/env bash

SBATCH=$false
while [ "$1" != "" ]; do
    PARAM=`echo $1 | awk -F= '{print $1}'`
    VALUE=`echo $1 | awk -F= '{print $2}'`
    case $PARAM in
        -h | --help)
            usage
            exit
            ;;
        --agents)
            AGENTS=$VALUE
            ;;
        --sbatch)
            SBATCH=$true
	    ;;
        *)
            echo "ERROR: unknown parameter \"$PARAM\""
            usage
            exit 1
            ;;
    esac
    shift
done

echo "agents is $AGENTS sbatch is $SBATCH";

for i in $(seq $AGENTS)
do
    echo "starting sbatch $i" &
    sbatch run_child.sbatch &
done
