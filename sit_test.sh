#!/bin/bash
export PYTHONPATH="/data/chenzhanni:$PYTHONPATH"

python main.py --test --name test \
        --force \
        --num_envs 4 \
        --cfg cfg/amp_sit_simple.yaml \
        --device 4 \
        --record \
        --ckpt logs/sit_simple/best.pth

 # --ckpt weights/humanvla.pth \