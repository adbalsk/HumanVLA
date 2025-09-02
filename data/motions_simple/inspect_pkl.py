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

if __name__ == "__main__":
    # 用法: python inspect_pkl.py your_file.pkl
    if len(sys.argv) != 2:
        print("Usage: python inspect_pkl.py <path_to_pkl>")
    else:
        inspect_pkl(sys.argv[1])