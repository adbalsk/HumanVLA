#!/bin/bash
export PYTHONPATH="/data/chenzhanni:$PYTHONPATH"

torchrun --standalone --nnode=1 --nproc_per_node=1 -m main --headless --cfg cfg/amp_sit.yaml --name sit --force --device 7

# python main.py --name sit \
#         --force --num_envs 4 \
#         --cfg cfg/amp_sit.yaml \
#         --device 3 

 # --ckpt weights/humanvla.pth \