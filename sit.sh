#!/bin/bash
export PYTHONPATH="/data/chenzhanni:$PYTHONPATH"

python main.py --test --name play \
        --force --num_envs 4 \
        --cfg cfg/amp_sit.yaml \
        --ckpt weights/humanvla.pth \
        --device 3 \
        --record