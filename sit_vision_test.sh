#!/bin/bash
export PYTHONPATH="/data/chenzhanni:$PYTHONPATH"

python main.py --test --name test \
        --force \
        --num_envs 16 \
        --cfg cfg/student_sit.yaml \
        --device 4 \
        --record \
        --ckpt logs/sit_vision/epoch_50000.pth

 # --ckpt weights/humanvla.pth \