#!/bin/bash
export PYTHONPATH="/data/chenzhanni:$PYTHONPATH"

python main.py --test --name sit_test \
        --force \
        --num_envs 4 \
        --cfg cfg/amp_sit_simple.yaml \
        --device 3 \
        --record \
        --ckpt logs/sit_train/epoch_20000.pth
        #--ckpt logs/sit_train/best.pth

 # --ckpt weights/humanvla.pth \