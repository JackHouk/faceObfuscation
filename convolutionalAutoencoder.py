# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 02:29:42 2018

@author: Jack
"""

import tensorflow as tf
import pickle
import numpy as np


#images are 112x112, this should be a flag
dims = 112

def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    input_layer = tf.reshape(features["x"], [-1, dims, dims, 1])
    labels = tf.reshape(labels, [-1, dims, dims, 1])
    print('input layer',input_layer.shape)
    
    # Convolutional Layer #1
    # Padding is added to preserve width and height.
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=512,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.tanh)
    print('conv 1', conv1.shape)

    # Convolutional Layer #2
    # Padding is added to preserve width and height.
    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=512,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.tanh)
    print('conv 2', conv2.shape)
    
    # Convolutional Layer #3
    conv3 = tf.layers.conv2d(
        inputs=conv2,
        filters=256,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.tanh)
    print('conv3', conv3.shape)

    # Convolutional Layer #4
    conv4 = tf.layers.conv2d(
        inputs=conv3,
        filters=128,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.tanh)
    print('conv4', conv4.shape)

    conv5 = tf.layers.conv2d(
        inputs=conv4,
        filters=256,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.tanh)
    print('conv5', conv5.shape)

    conv6 = tf.layers.conv2d(
        inputs=conv5,
        filters=128,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.tanh)
    print('conv6', conv6.shape)

    conv7 = tf.layers.conv2d(
        inputs=conv6,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.tanh)
    print('conv7', conv7.shape)
    
    conv8 = tf.layers.conv2d(
        inputs=conv7,
        filters=1,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.tanh)
    print('conv8', conv8.shape)
    

    # Normal loss function
    print('labels', labels.shape)
    loss = loss = tf.reduce_mean(tf.square(conv8 - labels))

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.000000001)
        train_op = optimizer.minimize(loss=loss,
                global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
        labels=labels, predictions=conv8)}
    return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(unused_argv):
    # Load training and eval data
    #train inputs
    file_open = open('train_x.pkl', 'rb')
    train_data = np.asarray(pickle.load(file_open), dtype=np.float32)
    file_open.close()
    
    #train targets
    file_open = open('train_t.pkl', 'rb')
    train_labels =np.asarray(pickle.load(file_open), dtype=np.float32)
    print(train_labels.shape)
    file_open.close()

    #test inputs
    file_open = open('test_x.pkl', 'rb')
    eval_data = np.asarray(pickle.load(file_open), dtype=np.float32)
    file_open.close()
    
    #test targets
    file_open = open('test_t.pkl', 'rb')
    eval_labels = np.asarray(pickle.load(file_open), dtype=np.float32)
    print(eval_labels.shape)
    file_open.close()

    # Create the Estimator
    facial_classifier = tf.estimator.Estimator(
            model_fn=cnn_model_fn, model_dir="tensorflow/models/facial_cnn_model3")

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {}#"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": train_data},
            y=train_labels,
            batch_size=10,
            num_epochs=45,
            shuffle=True)
    facial_classifier.train(
            input_fn=train_input_fn,
            steps=2000,
            hooks=[logging_hook])

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": eval_data[30, :, :]},
            y=eval_labels[30, :, :],
            num_epochs=1,
            shuffle=False)
    eval_results = facial_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)

if __name__ == "__main__":
        tf.app.run()