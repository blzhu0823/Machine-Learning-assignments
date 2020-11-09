#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2020/11/8 11:31 上午
# @Author  : zbl
# @Email   : funnyzhu1997@gmail.com
# @File    : models.py
# @Software: PyCharm


import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_class, drop_out=0.2):

        super(MLPNN, self).__init__()
        self.mlp = nn.ModuleList([nn.Linear(input_size, hidden_sizes[0])] +
                                 [nn.Linear(hidden_sizes[i - 1], hidden_sizes[i])
                                  for i in range(1, len(hidden_sizes))])
        self.project = nn.Linear(hidden_sizes[-1], num_class)
        self.dropout = nn.Dropout(drop_out)

    def forward(self, x):
        """

        :param x:  (batch, input_size) -> (batch, num_class)
        """
        for linear in self.mlp:
            x = self.dropout(F.relu(linear(x)))
        logits = self.project(x)
        return F.softmax(logits, dim=-1)


if __name__ == '__main__':
    model = MLPNN(10, [20, 40, 30], 5)
    x = torch.randn(32, 10)
    print(model(x).shape)