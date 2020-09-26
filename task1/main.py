import torch
import torchvision
from torchvision import datasets, transforms
import numpy as np


train_dataset = datasets.MNIST(root='./dataset', 
                train=True,
                transform=transforms.ToTensor(),
                download=False)

test_dataset = datasets.MNIST(root='./dataset',
               train=False,
               transform=transforms.ToTensor(),
               download=False)
               
#print(train_dataset)
#print(test_dataset)
train_images = np.array([item[0].numpy() for item in train_dataset]).reshape(len(train_dataset), -1)
train_labels = np.array([item[1] for item in train_dataset])
test_images  = np.array([item[0].numpy() for item in test_dataset]).reshape(len(test_dataset), -1)
test_labels  = np.array([item[1] for item in test_dataset])


print(len(train_images), len(train_labels), len(test_images), len(test_labels))
print(train_images.shape, train_labels.shape, test_images.shape, test_labels.shape)

print(np.max(train_images), np.min(train_images))


def KNN(id, k, images=train_images, labels=train_labels, distance='Euclidean'):
    if distance == 'Euclidean':
        d = np.sqrt(np.sum((images - test_images[id])**2, axis=1))
    else:
        pass
    ix = np.argsort(d)
    countDict = {}
    for i in range(k):
        label = labels[ix[i]]
        countDict[label] = countDict.get(label, 0) + 1
    result = sorted(countDict.items(), key=lambda x:x[1], reverse=True)
    return result[0][0]

correction = 0
for i in range(len(test_images)//10):
    if KNN(i, 9) == test_labels[i]:
        correction += 1
print('accuracy:', str(100*correction/((len(test_images)//10))) + '%')