import torch

# 4. 加载使用（无需原结构文件）
loaded_model = torch.jit.load("/root/volume/ctc/train/work_dir/avg_10.pt")
print(loaded_model)