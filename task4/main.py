#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2020/10/18 7:23 下午
# @Author  : zbl
# @Email   : funnyzhu1997@gmail.com
# @File    : main.py
# @Software: PyCharm


from torchvision import datasets, transforms
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np





def least_squard_regression(x, y, lamda):
    x_t = x.T
    x_t_dot_x = x_t.dot(x)
    temp = x_t_dot_x + lamda*np.identity(x.shape[-1])
    temp_inverse = np.linalg.inv(temp)
    w = temp_inverse.dot(x.T).dot(y)
    return w


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
    print('x_train:', x_train.shape)
    print('y_train', y_train.shape)
    print('x_test:', x_test.shape)
    print('y_test:', y_test.shape)

w = least_squard_regression(x_train, y_train, 0.2)
print(w.shape)
acc1 = np.sum(np.around(w.dot(x_test.T)) == y_test) / len(y_test)
print('regularized lsr acc:', acc1)





lda = LinearDiscriminantAnalysis(n_components=3)
lda.fit(x_train, y_train)
y_preds = lda.predict(x_test)
acc2 = np.sum((y_preds == y_test)) / len(y_test)
fig = plt.figure()
ax = Axes3D(fig)
x_new = lda.transform(x_test)
ax.scatter(x_new[:, 0], x_new[:, 1], x_new[:, 2], marker='o', c=y_test)
plt.show()
print('fda acc:', acc2)
