#!/bin/bash
export PYTHONPATH="/data/chenzhanni:$PYTHONPATH"

python main.py --name sit \
        --force --num_envs 4 \
        --cfg cfg/amp_sit.yaml \
        --device 3 \
        --record

 # --ckpt weights/humanvla.pth \