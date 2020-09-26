#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2020/9/26 10:38 下午
# @Author  : zbl
# @Email   : funnyzhu1997@gmail.com
# @File    : model.py
# @Software: PyCharm


import torch.nn as nn
import torch
from torch.nn.functional import max_pool2d, relu


class Model(nn.Module):

    def __init__(self, channel_in, num_of_class=10, filters=[30, 50], kernel_size=[5, 3], dropout_rate=0.3):

        super(Model, self).__init__()
        channel_ins = [channel_in] + filters[:-1]
        self.convs = [nn.Conv2d(channel_ins[i], filters[i], kernel_size[i])
                      for i in range(len(filters))]
        self.dropout = nn.Dropout(dropout_rate)
        self.feature2label = nn.Linear(5 * 5 * filters[-1], num_of_class)

    def forward(self, X: torch.Tensor) -> object:
        for conv in self.convs:
            X = relu(conv(X))
            X = max_pool2d(X, 2)

        X = X.view(X.shape[0], -1)
        X = self.dropout(X)
        logits = self.feature2label(X)
        return logits


if __name__ == '__main__':
    X = torch.randn(3, 28, 28)
    model = Model(channel_in=1)
    print(model(X))