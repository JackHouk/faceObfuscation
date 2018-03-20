# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 11:46:52 2018

@author: Jack
"""

import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt

# load the model here
sess=tf.Session()    
saver = tf.train.import_meta_graph('tensorflow/denoise_relu/model.ckpt-751.meta')
saver.restore(sess,tf.train.latest_checkpoint('tensorflow\\denoise_relu'))
graph = tf.get_default_graph()


# populate the x and y values to make the predictions here
y_pred = graph.get_tensor_by_name('Reshape_1:0')

file_open = open('0000/M3/train/full_set.pkl', 'rb')
train_data = pickle.load(file_open)
file_open.close()

train_flat = [value for row in train_data[149] for value in row]
train_data = np.asarray(train_flat, dtype=np.float32)
train_data = train_data.reshape([1, 13225])

x = graph.get_tensor_by_name('Reshape:0')
y = graph.get_tensor_by_name('y:0')

feed_dict = {x:train_data, y:np.zeros([1, 115, 115], np.float32)}

results = sess.run(y_pred, feed_dict).tolist()
plt.imshow(results[0], cmap='gray')
plt.show()