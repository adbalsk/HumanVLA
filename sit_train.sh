#!/bin/bash
export PYTHONPATH="/data/chenzhanni:$PYTHONPATH"

torchrun --standalone --nnode=1 --nproc_per_node=1 -m main --headless --cfg cfg/amp_sit.yaml --name sit_train --force --device 6 \
        --num_envs 4096

# python main.py --name sit \
#         --force --num_envs 4 \
#         --cfg cfg/amp_sit.yaml \
#         --device 3 

 # --ckpt weights/humanvla.pth \