import os
import yaml

motion_dir = "./"
files = sorted([f for f in os.listdir(motion_dir) if f.endswith(".pkl")])

# 构造 motion.yaml 的数据结构
motion_list = [[os.path.join(motion_dir, f), 1] for f in files]

with open("motion.yaml", "w") as f:
    yaml.dump(motion_list, f, default_flow_style=False, sort_keys=False)

print("✅ motion.yaml 已生成，共包含", len(motion_list), "个文件")