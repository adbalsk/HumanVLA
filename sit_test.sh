#!/bin/bash
export PYTHONPATH="/data/chenzhanni:$PYTHONPATH"

python main.py --test --name test \
        --force \
        --num_envs 16 \
        --cfg cfg/amp_sit_obs.yaml \
        --device 4 \
        --record \
        --ckpt logs/sit_init/best.pth

 # --ckpt weights/humanvla.pth \