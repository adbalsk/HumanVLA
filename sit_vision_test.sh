#!/bin/bash
export PYTHONPATH="/data/chenzhanni:$PYTHONPATH"
#export CUDA_VISIBLE_DEVICES=0

python main.py --test --name test \
        --force \
        --num_envs 20 \
        --cfg cfg/student_sit.yaml \
        --device 0 \
        --ckpt logs/sit_vision/epoch_54000.pth \
        --record

 # --ckpt weights/humanvla.pth \