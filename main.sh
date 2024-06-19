#!/bin/bash
# Ask for parameters in command line
source1=$1
source2=$2
# Launch the first Python script in the background
python new_inference.py --video_source="$source1" &

# Launch the second Python script in the background
python new_inference.py --video_source="$source2" &

# Kill all processes when ctrl+c
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT
wait