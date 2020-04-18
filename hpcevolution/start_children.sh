#!/usr/bin/env bash

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
        *)
            echo "ERROR: unknown parameter \"$PARAM\""
            usage
            exit 1
            ;;
    esac
    shift
done

echo "agents is $AGENTS";

for i in $(seq $AGENTS)
do
    echo "starting $i" &
    nohup  bash start_child.sh $i &
done