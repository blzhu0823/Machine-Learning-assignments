#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2020/11/8 11:30 上午
# @Author  : zbl
# @Email   : funnyzhu1997@gmail.com
# @File    : main.py
# @Software: PyCharm

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import get_loader
from models import MLPNN
from torch.optim import Adam, SGD
from torch.nn import CrossEntropyLoss, MSELoss



DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCH = 10
BATCH_SIZE = 64
NUM_CLASSES = 10
HIDDEN_SIZES = [512, 512]
LR = 0.01

def run_epoch(model, data_loader, loss_func, optimizer, eval=False, device='cpu', loss_name='crossentropy'):
    if eval:
        model.eval()
    else:
        model.train()
    total_loss = 0.0
    total_len = 0
    total_acc = 0
    if not eval:
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            if loss_name == 'mse':
                y_one_hot = F.one_hot(y, num_classes=NUM_CLASSES).float()
            out = model(x)
            if loss_name == 'mse':
                loss = loss_func(out, y_one_hot)
            else:
                loss = loss_func(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss*len(y)
            total_len += len(y)
            pred = torch.argmax(out, dim=-1)
            total_acc += (pred == y).sum()
    else:
        with torch.no_grad():
            for x, y in data_loader:
                x, y = x.to(device), y.to(device)
                if loss_name == 'mse':
                    y_one_hot = F.one_hot(y, num_classes=NUM_CLASSES).float()
                out = model(x)
                if loss_name == 'mse':
                    loss = loss_func(out, y_one_hot)
                else:
                    loss = loss_func(out, y)
                total_loss += loss*len(y)
                total_len += len(y)
                pred = torch.argmax(out, dim=-1)
                total_acc += (pred == y).sum()

    return total_loss/len(data_loader.dataset), total_acc/total_len



if __name__ == '__main__':
    loss_name = 'cross_entropy' # ['cross_entropy', 'mse']
    train_dataloader, test_dataloader = get_loader('../dataset/MNIST/processed', BATCH_SIZE)
    model = MLPNN(784, HIDDEN_SIZES, NUM_CLASSES, drop_out=0).to(DEVICE)
    train_loss_history = []
    train_acc_history = []
    test_loss_history = []
    test_acc_history = []
    if loss_name == 'cross_entropy':
        loss_func = CrossEntropyLoss()
    elif loss_name == 'mse':
        loss_func = MSELoss()
    optimizer = SGD(model.parameters(), lr=LR)
    for e in tqdm(range(EPOCH)):
        train_loss, train_acc = run_epoch(model, train_dataloader, loss_func, optimizer, False, DEVICE, loss_name)
        test_loss, test_acc = run_epoch(model, test_dataloader, loss_func, optimizer, True, DEVICE, loss_name)
        tqdm.write('epoch {}, train loss {}, acc{}'.format(e+1, train_loss, train_acc))
        tqdm.write('epoch {}, test loss {}, acc{}'.format(e+1, test_loss, test_acc))
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)
        test_loss_history.append(test_loss)
        test_acc_history.append(test_acc)

    fig = plt.figure(figsize=(20, 10))
    fig.suptitle('results for CrossEntropy loss')
    plt.subplot(1, 2, 1)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(range(1, EPOCH+1), train_loss_history, 'r--', label='train_loss')
    plt.plot(range(1, EPOCH+1), test_loss_history, 'b--', label='test_loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.plot(range(1, EPOCH+1), train_acc_history, 'r--', label='train_acc')
    plt.plot(range(1, EPOCH+1), test_acc_history, 'b--', label='test_acc')
    plt.legend()
    plt.savefig('./fig/fig2.png')