#!/bin/bash
export PYTHONPATH="/data/chenzhanni:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=3

python main.py --test --name test \
        --force \
        --num_envs 12 \
        --cfg cfg/student_sit.yaml \
        --device 0 \
        --record \
        --ckpt logs/sit_vision/epoch_29000.pth

 # --ckpt weights/humanvla.pth \