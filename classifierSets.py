# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 13:16:04 2018

@author: Jack
"""

import pickle
from PIL import Image
from glob import glob
from random import shuffle

size = 112

imgs = glob('Data/MIT-CBCL/data/all/train/M0/*.jpg')
labels = [0]*len(imgs)
for i in range(len(labels)):
    if i < 150:
        labels[i] = 0
    elif i < 301:
        labels[i] = 1
    elif i < 453:
        labels[i] = 2
    elif i < 603:
        labels[i] = 3
    elif i < 755:
        labels[i] = 4
    elif i < 907:
        labels[i] = 5
    elif i < 1056:
        labels[i] = 6
    elif i < 1208:
        labels[i] = 7
    else:
        labels[i] = 8
        
indexes = list(range(len(labels)))
shuffle(indexes)

train = indexes[:1100]
test = indexes[1100:]

train_x = []
train_t = []

test_x = []
test_t = []

for i in train:
    train_t.append(labels[i])
    with Image.open(imgs[i]) as image:
        image = image.resize((size, size))
        dat = list(image.getdata())
        dat = [dat[offset:offset+size] for offset in range(0, size*size, size)]
        for row in range(len(dat)):
            for col in range(len(dat[row])):
                dat[row][col] = dat[row][col][0]
        train_x.append(dat)
        
for i in test:
    test_t.append(labels[i])
    with Image.open(imgs[i]) as image:
        image = image.resize((size, size))
        dat = list(image.getdata())
        dat = [dat[offset:offset+size] for offset in range(0, size*size, size)]
        for row in range(len(dat)):
            for col in range(len(dat[row])):
                dat[row][col] = dat[row][col][0]
        test_x.append(dat)
        
with open('train_x.pkl', 'wb') as f:
    pickle.dump(train_x, f)

with open('test_x.pkl', 'wb') as f:
    pickle.dump(test_x, f)
    
with open('train_t.pkl', 'wb') as f:
    pickle.dump(train_t, f)
    
with open('test_t.pkl', 'wb') as f:
    pickle.dump(test_t, f)