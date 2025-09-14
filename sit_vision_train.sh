#!/bin/bash
export PYTHONPATH="/data/chenzhanni:$PYTHONPATH"

# torchrun --standalone --nnode=1 --nproc_per_node=1 -m main --headless --cfg cfg/amp_sit.yaml --name sit_train --force --device 6 \
#         --num_envs 4096

python main.py --name sit_vision \
        --force --num_envs 512 \
        --cfg cfg/student_sit.yaml \
        --device 6 --headless \
        --ckpt logs/sit_vision0913/epoch_50000.pth

# python main.py --name sit \
#         --force --num_envs 4 \
#         --cfg cfg/amp_sit.yaml \
#         --device 3 

 # --ckpt weights/humanvla.pth \