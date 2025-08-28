import os
import yaml

motion_dir = "./data/motions2"
files = sorted([f for f in os.listdir(motion_dir) if f.endswith(".pkl")])

# 构造 motion.yaml 的数据结构
motion_list = []
for f in files:
    if f.startswith("SAMP_chair"): w = 10
    elif f.startswith("SAMP"): w = 2
    else: w = 1
    motion_list.append([os.path.join(motion_dir, f), w])

with open(os.path.join(motion_dir, "motion.yaml"), "w") as f:
    yaml.dump(motion_list, f, default_flow_style=False, sort_keys=False)

print("motion.yaml 已生成，共包含", len(motion_list), "个文件")