# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 00:20:48 2018

@author: Jack
"""

import pickle
from PIL import Image
from glob import glob
from random import shuffle

imgs = glob('Data/raisrResults/M5/*.jpg')
for i in range(len(imgs)):
    imgs[i] = imgs[i][imgs[i].find('\\') + 1:]
    imgs[i] = imgs[i][:imgs[i].find('_')]
    
shuffle(imgs)
train = imgs[:400]
test = imgs[400:]

train_x = []
train_t = []

test_x = []
test_t = []

size = 112

for im in train:
    if im == 150 or im == 318 or im == 301 or im == 66:
        continue
    with Image.open('RAISR/raisr-master/train/' + str(im) + '.jpg') as image:
        image = image.resize((size, size))
        dat = list(image.getdata())
        dat = [dat[offset:offset+size] for offset in range(0, size*size, size)]
        for row in range(len(dat)):
            for col in range(len(dat[row])):
                dat[row][col] = dat[row][col][0]
        train_x.append(dat)
        
    with Image.open('RAISR/raisr-master/test/R0/' + str(im) + '.jpg') as image:
        image = image.resize((size, size))
        dat = list(image.getdata())
        dat = [dat[offset:offset+size] for offset in range(0, size*size, size)]
        for row in range(len(dat)):
            for col in range(len(dat[row])):
                dat[row][col] = dat[row][col][0]
        print(len(dat))
        print(len(dat[0]))
        train_t.append(dat)
        
for im in test:
    with Image.open('RAISR/raisr-master/train/' + str(im) + '.jpg') as image:
        image = image.resize((size, size))
        dat = list(image.getdata())
        dat = [dat[offset:offset+size] for offset in range(0, size*size, size)]
        for row in range(len(dat)):
            for col in range(len(dat[row])):
                dat[row][col] = dat[row][col][0]
        test_x.append(dat)
        
    with Image.open('RAISR/raisr-master/test/R0/' + str(im) + '.jpg') as image:
        image = image.resize((size, size))
        dat = list(image.getdata())
        dat = [dat[offset:offset+size] for offset in range(0, size*size, size)]
        for row in range(len(dat)):
            for col in range(len(dat[row])):
                dat[row][col] = dat[row][col][0]
        test_t.append(dat)
        
with open('train_x.pkl', 'wb') as f:
    pickle.dump(train_x, f)

with open('test_x.pkl', 'wb') as f:
    pickle.dump(test_x, f)
    
with open('train_t.pkl', 'wb') as f:
    pickle.dump(train_t, f)
    
with open('test_t.pkl', 'wb') as f:
    pickle.dump(test_t, f)