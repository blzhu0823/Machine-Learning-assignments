#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2020/9/26 10:20 下午
# @Author  : zbl
# @Email   : funnyzhu1997@gmail.com
# @File    : main.py
# @Software: PyCharm


import torch
import torchvision
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from model import Model

if __name__ == '__main__':
    epoch_num = 10
    learning_rate = 0.005



    train_dataset = datasets.MNIST(root='./dataset',
                                   train=True,
                                   transform=transforms.ToTensor(),
                                   download=False)

    test_dataset = datasets.MNIST(root='./dataset',
                                  train=False,
                                  transform=transforms.ToTensor(),
                                  download=False)

    train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=128)
    test_dataloader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=128)

    model = Model(channel_in=1)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fun = CrossEntropyLoss()

    for e in range(epoch_num):
        model.train()
        for X, y in train_dataloader:
            optimizer.zero_grad()
            logits = model(X)
            loss = loss_fun(logits, y)
            loss.backward()
            optimizer.step()

        model.eval()
        test_accs = []
        for X, y in test_dataset:
            X.unsqueeze_(1)
            logits = model(X)
            preds = torch.argmax(logits, dim=-1)
            acc = torch.mean((preds == y).float())
            test_accs.append(acc)

        print('epoch {} train acc: {}%'.format(e, 100*sum(test_accs)/len(test_accs)))