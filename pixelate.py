# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 01:31:32 2018

@author: Jack
"""


from glob import glob
from math import floor
from PIL import Image
from random import shuffle
from shutil import rmtree
import os
import pickle

#cleanup for multiple runs
toRm = glob('**/M*')
for remove in toRm:
    rmtree(os.path.join(remove))

imgs = []
train = []
test = []

for i in range(10):
    imgs = glob('000' + str(i) + '/*.jpg')
    shuffle(imgs)
    train = train + imgs[:150]
    test = test + imgs[150:]
    
    for im in train[i * 150: i * 150 + 150]:
        image = Image.open(im)
        width, height = image.size
        for pixelSize in range(1, 9):
            image = image.resize((floor(width/pixelSize), floor(height/pixelSize)), Image.NEAREST)
            image = image.resize((width, height), Image.NEAREST)
            
            if not os.path.isdir('000' + str(i) + '/M' + str(pixelSize - 1)):
                os.mkdir('000' + str(i) + '/M' + str(pixelSize - 1))
            if not os.path.isdir('000' + str(i) + '/M' + str(pixelSize - 1) + '/train/'):
                os.mkdir('000' + str(i) + '/M' + str(pixelSize - 1) + '/train/')
            
            image.save('000' + str(i) + '/M' + str(pixelSize - 1) + '/train/' + im[10:])
        
        image.close()

    for im in test[i * 50: i * 50 + 50]:
        image = Image.open(im)
        width, height = image.size
        for pixelSize in range(1, 9):
            image = image.resize((floor(width/pixelSize), floor(height/pixelSize)), Image.NEAREST)
            image = image.resize((width, height), Image.NEAREST)
            
            if not os.path.isdir('000' + str(i) + '/M' + str(pixelSize - 1)):
                os.mkdir('000' + str(i) + '/M' + str(pixelSize - 1))
            if not os.path.isdir('000' + str(i) + '/M' + str(pixelSize - 1) + '/test/'):
                os.mkdir('000' + str(i) + '/M' + str(pixelSize - 1) + '/test/')
            
            image.save('000' + str(i) + '/M' + str(pixelSize - 1) + '/test/' + im[10:])
        
        image.close()
        
for i in range(10):
    base_path = '000' + str(i) + '/M'
    train_names = train[i * 150: i * 150 + 150]
    train_names = [name[10:] for name in train_names]
    test_names = test[i * 50: i * 50 + 50]
    test_names = [name[10:] for name in test_names]
    for difficulty in range(8):
        path = base_path + str(difficulty)
        train_set = []
        test_set = []
        for im in train_names:
            image = Image.open(path + '/train/' + im)
            dat = list(image.getdata())
            dat = [dat[offset:offset+width] for offset in range(0, width*height, width)]
            for row in range(len(dat)):
                for col in range(len(dat[row])):
                    dat[row][col] = dat[row][col][0]
            train_set.append(dat)
            image.close()
            
        for im in test_names:
            image = Image.open(path + '/test/' + im)
            dat = list(image.getdata())
            dat = [dat[offset:offset+width] for offset in range(0, width*height, width)]
            for row in range(len(dat)):
                for col in range(len(dat[row])):
                    dat[row][col] = dat[row][col][0]
            test_set.append(dat)
            image.close()

        train_pkl_f = open(path + '/train/full_set.pkl', 'wb')            
        test_pkl_f = open(path + '/test/full_set.pkl', 'wb')
        pickle.dump(train_set, train_pkl_f)
        pickle.dump(test_set, test_pkl_f)
        train_pkl_f.close()
        test_pkl_f.close()