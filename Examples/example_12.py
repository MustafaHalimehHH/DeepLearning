# https://www.kaggle.com/karthikreddy25/vgg-inspired-architecture-99-recall
import os
import sys
import random
import numpy
import pandas
import keras
import cv2
import matplotlib.pyplot as plt


# _URL = 'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz'
data_dir = 'D:\Halimeh\Datasets\Kaggle\chest-xray-pneumonia\chest_xray'
normal_train = list(os.listdir(os.path.join(data_dir, 'train', 'NORMAL')))
pneumonia_train = list(os.listdir(os.path.join(data_dir, 'train', 'PNEUMONIA')))
print('normal_train', len(normal_train))
print('pneumonia_train', len(pneumonia_train))


def display(classes, cols, read_as_rgb=True, cmap=None):
    for _class in classes:
        fig, axes = plt.subplots(nrows=1, ncols=cols, figsize=(14, 10), squeeze=False)
        fig.tight_layout()
        for l in range(1):
            for m, img in enumerate(classes[_class]):
                axes[l][m].imshow(img, cmap=cmap)


def preprocess(image, input_mode='gray_scale', reshape=True):
    print('image', image, type(image))
    img = image
    if input_mode == 'rgb':
        B, G, R = cv2.split(img)
        R = cv2.equalizeHist(R)
        B = cv2.equalizeHist(B)
        G = cv2.equalizeHist(G)
        img = cv2.merge([B, G, R])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    elif input_mode == 'gray_scale':
        img = cv2.equalizeHist(img)
        if reshape:
            img = img.reshape(224, 224, 1)
    img = img / 255.0
    return img


sample_positive = random.sample(pneumonia_train, 3)
print(sample_positive)
classes = {'Original': [cv2.imread(os.path.join(data_dir, 'train', 'pneumonia', img), 0) for img in sample_positive],
           'PreProcessed': [preprocess(cv2.imread(os.path.join(data_dir, 'train', 'pneumonia', img), 0), input_mode='gray_scale', reshape=False) for img in sample_positive]}
display(classes, 3, cmap='gray')
plt.show()