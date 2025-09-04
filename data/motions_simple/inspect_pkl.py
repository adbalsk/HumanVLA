import pickle as pkl
import torch
import numpy as np
import sys

def inspect_pkl(path):
    with open(path, "rb") as f:
        data = pkl.load(f)

    print(f"✅ Keys in {path}:")
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:20s} -> torch.Tensor, shape = {tuple(v.shape)}, dtype = {v.dtype}")
        elif isinstance(v, np.ndarray):
            print(f"  {k:20s} -> numpy.ndarray, shape = {v.shape}, dtype = {v.dtype}")
        else:
            print(f"  {k:20s} -> {type(v)}")
    
    last_idx = -1  

    last_pose = {
        "dof_pos": data["dof_pos"][last_idx],          # (28,)
        "dof_vel": data["dof_vel"][last_idx],          # (28,)
        "rigid_body_pos": data["rigid_body_pos"][last_idx],  # (15, 3)
        "rigid_body_rot": data["rigid_body_rot"][last_idx],  # (15, 4)
        "rigid_body_vel": data["rigid_body_vel"][last_idx],  # (15, 3)
        "rigid_body_anv": data["rigid_body_anv"][last_idx],  # (15, 3)
        "object_pos": data["object_pos"][last_idx],    # (3,)
        "object_vel": data["object_vel"][last_idx],    # (3,)
    }

    print(last_pose)

if __name__ == "__main__":
    # 用法: python inspect_pkl.py your_file.pkl
    if len(sys.argv) != 2:
        print("Usage: python inspect_pkl.py <path_to_pkl>")
    else:
        inspect_pkl(sys.argv[1])