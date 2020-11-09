#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2020/11/8 11:30 上午
# @Author  : zbl
# @Email   : funnyzhu1997@gmail.com
# @File    : utils.py
# @Software: PyCharm


from torch.utils.data import Dataset, DataLoader
import os
import torch


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.len = len(data[0])

    def __getitem__(self, item):
        return self.data[0][item], self.data[1][item]

    def __len__(self):
        return self.len

def my_colla_fn(batch):
    """

    :param data: a list of items whose len is batchsize
    """
    x = torch.cat([item[0].reshape(1, -1) for item in batch], dim=0).float()
    y = torch.LongTensor([item[1] for item in batch])
    return x, y




def get_loader(path, batch_size):
    train_dataset = MyDataset(torch.load(os.path.join(path, 'training.pt')))
    test_dataset = MyDataset(torch.load(os.path.join(path, 'test.pt')))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  collate_fn=my_colla_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=my_colla_fn)
    return train_dataloader, test_dataloader


if __name__ == '__main__':
    train_dataloader, test_dataloader = get_loader('../dataset/MNIST/processed', 32)
    for batch in test_dataloader:
        print(batch[0].shape, batch[1].shape)