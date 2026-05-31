# training_config.py
# 训练配置参数整理，提取自FedAvg.py

import torch
import sys
import os

# 设备设置
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ALEXNET_MAX_SPLIT_POINT = 8  # AlexNet模型的最大分割点
VGG11_MAX_SPLIT_POINT = 11  # VGG11模型的最大分割点
VGG16_MAX_SPLIT_POINT = 16  # VGG16模型的最大分割点
RESNET18_MAX_SPLIT_POINT = 18  # ResNet18模型的最大分割点
# ResNet18可分割点（避免在残差块内部切分）
# 每个BasicBlock由两段组成，只允许在完整块边界切分。
# 1 表示仅stem在lower端，3/5/.../17 表示每次增加完整block。
RESNET18_VALID_SPLIT_POINTS = [1, 3, 5, 7, 9, 11, 13, 15, 17]