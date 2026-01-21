# kaldi2onnx.py
import re
import numpy as np
import torch
import torch.nn as nn

loaded_full_model = torch.load("/root/volume/ctc/train/my_model_full.pt")
print(loaded_full_model)