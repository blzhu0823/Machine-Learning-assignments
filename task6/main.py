#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2020/11/1 12:59 下午
# @Author  : zbl
# @Email   : funnyzhu1997@gmail.com
# @File    : main.py.py
# @Software: PyCharm


from sklearn import svm
from torchvision import datasets, transforms
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np


if __name__ == '__main__':
    train_dataset = datasets.MNIST(root='./dataset',
                                   train=True,
                                   transform=transforms.ToTensor(),
                                   download=True)

    test_dataset = datasets.MNIST(root='./dataset',
                                  train=False,
                                  transform=transforms.ToTensor(),
                                  download=True)

    x_train = [train_dataset[i][0].numpy() for i in range(len(train_dataset))]
    y_train = [train_dataset[i][1] for i in range(len(train_dataset))]
    x_test = [test_dataset[i][0].numpy() for i in range(len(test_dataset))]
    y_test = [test_dataset[i][1] for i in range(len(test_dataset))]

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)
    print('train x shape:', x_train.shape)
    print('train y shape:', y_train.shape)
    print('test x shape:', x_test.shape)
    print('test y shape:', y_test.shape)

    # svm_clf = svm.SVC(C=1.0, kernel='rbf')
    # svm_clf.fit(x_train, y_train)
    svm_clf = make_pipeline(StandardScaler(), svm.SVC(C=1.0, kernel='linear'))
    svm_clf.fit(x_train, y_train)
    print('train acc:', svm_clf.score(x_train, y_train))
    print('test acc:', svm_clf.score(x_test, y_test))