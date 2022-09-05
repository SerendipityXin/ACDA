# coding=utf-8

import sys
from torch import nn
sys.path.append('/data/meiqi.hu/PycharmProjects/HyperspectralACD/MyFunction/')

# 自动编码器--无监督算法
'''
该网络的目的在于重构输入，使其隐藏层的数据表示更加好，
利用了反向传播，将输入值和目标值设置一样，就可以不断的进行迭代训练。
in_dim:
hid_dim1:
hid_dim2:
'''


# 多层感知机-、
class AutoEncoder(nn.Module):
    def __init__(self, in_dim, hid_dim1, hid_dim2):
        super(AutoEncoder, self).__init__()
        # 输出维度
        self.out_dim = in_dim
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hid_dim1, bias=True),
            nn.ReLU(),
            nn.Linear(hid_dim1, hid_dim2, bias=True),
            nn.ReLU(),
        )
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(hid_dim2, hid_dim1, bias=True),
            nn.ReLU(),
            nn.Linear(hid_dim1, in_dim, bias=True),
            nn.ReLU(),
        )

    # 前向传播
    def forward(self, x):
        feature = self.encoder(x)
        cons_x = self.decoder(feature)
        return cons_x
