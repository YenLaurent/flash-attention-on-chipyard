"""
将C端单头Flash Attention Forward的输出与Pytorch标准实现进行对比验证的脚本
得到相关误差指标并进行可视化
"""
import torch
from torch import nn
from flash_attention_forward_single_head_eval import flash_attention_forward_single_head
import numpy as np
import matplotlib.pyplot as plt

SEQ_LEN = 510
HEAD_DIM = 64
BR = 128
BC = 128
SEED = 42


