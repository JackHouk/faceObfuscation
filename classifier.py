# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 13:05:28 2018

@author: Jack
"""

import tensorflow as tf
import pickle
import numpy as np

dims = 112

tf.logging.set_verbosity(tf.logging.INFO)
#images are 112x112, this should be a flag

def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    input_layer = tf.reshape(features["x"], [-1, dims, dims, 1])
    #print('input layer',input_layer.shape)
   # flat = tf.reshape(input_layer, -1, dims * dims * 10)
    # Convolutional Layer #1
    # Padding is added to preserve width and height.
    labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=9)
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[10, 16],
        padding="same",
        activation=tf.nn.relu)
    print('conv 1', conv1.shape)

    # Convolutional Layer #2
    # Computes 64 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 48, 48, 36]
    # Output Tensor Shape: [batch_size, 48, 48, 36]
    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=32,
        kernel_size=[6, 6],
        padding="same",
        activation=tf.nn.relu)
    print('conv 2', conv2.shape)

    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 48, 48, 36]
    # Output Tensor Shape: [batch_size, 24, 24, 36]
    pool1 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    print('pool 1', pool1.shape)

    # Convolutional Layer #3
    # Computes 128 features using a 3x3 filter and a stride of __
    # Input Tensor Shape: [batch_size, 24, 24, 24]
    # Output Tensor Shape: [batch_size, 24, 24, 64]
    conv3 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[6, 6],
        padding='same',
        activation=tf.nn.relu)
    print('conv3', conv3.shape)

    # Convolutional Layer #4
    # Computes 128 features using a 3x3 filter and a stride of __
    # Input Tensor Shape: [batch_size, 24, 24, 36]
    # Output Tensor Shape: [batch_size, 24, 24, 64]
    conv4 = tf.layers.conv2d(
        inputs=conv3,
        filters=64,
        kernel_size=[6, 6],
        padding='same',
        activation=tf.nn.relu)
    print('conv4', conv4.shape)

    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 24, 24, 64]
    # Output Tensor Shape: [batch_size, 12, 12, 64]
    pool2 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)
    print('pool 2', pool2.shape)

    # Convolutional Layer #3
    # Computes 128 features using a 3x3 filter and a stride of __
    # Input Tensor Shape: [batch_size, 12, 12, 128]
    # Output Tensor Shape: [batch_size, 12, 12, 128]
    conv5 = tf.layers.conv2d(
        inputs=pool2,
        filters=128,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu)
    print('conv5', conv5.shape)

    # Convolutional Layer #3
    # Computes 128 features using a 3x3 filter and a stride of __
    # Input Tensor Shape: [batch_size, 12, 12, 128]
    # Output Tensor Shape: [batch_size, 12, 12, 128]
    #conv6 = tf.layers.conv2d(
    #    inputs=conv5,
    #    filters=128,
    #    kernel_size=[3, 3],
    #    padding='same',
    #    activation=tf.nn.relu)
    #print('conv6', conv6.shape)

    # Pooling Layer #3
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 12, 12, 128]
    # Output Tensor Shape: [batch_size, 6, 6, 128]
    pool3 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[2, 2], strides=2)
    print('pool 3', pool3.shape)

    # Convolutional Layer #3
    # Computes 128 features using a 3x3 filter and a stride of __
    # Input Tensor Shape: [batch_size, 6, 6, 128]
    # Output Tensor Shape: [batch_size, 6, 6, 256]
    conv7 = tf.layers.conv2d(
        inputs=pool3,
        filters=128,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu)
    print('conv7', conv7.shape)

    # Pooling Layer #3
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 6, 6, 128]
    # Output Tensor Shape: [batch_size, 3, 3, 128]
    pool4 = tf.layers.max_pooling2d(inputs=conv7, pool_size=[2, 2], strides=2)
    print('pool 4', pool4.shape)

    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 3, 3, 128]
    # Output Tensor Shape: [batch_size, 3 * 3 * 128]
    pool_flat = tf.reshape(pool4, [-1, 7*7*128])
    print('pool flatten', pool_flat.shape)

    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 6 * 6 * 64]
    # Output Tensor Shape: [batch_size, 1024]
    dense = tf.layers.dense(inputs=pool_flat, units=512, activation=tf.nn.relu)

    # Add dropout operation; 0.6 probability that element will be kept
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    # Logits layer
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, 10]
    logits = tf.layers.dense(
        inputs=dropout,
        units = 9,
        activation=tf.nn.relu)

    
    predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            "classes": tf.identity(labels, name="classes"),
            # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
            # `logging_hook`.
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
            }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.softmax_cross_entropy(labels, logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0000001)
        train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
        labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(unused_argv):
    with tf.device('/gpu:0'):
        # Load training and eval data
        file_open = open('test_x.pkl', 'rb')
        train_data = np.asarray(pickle.load(file_open), dtype=np.float32)
        file_open.close()

        file_open = open('test_t.pkl', 'rb')
        train_labels =np.asarray(pickle.load(file_open), dtype=np.float32)
        file_open.close()

        file_open = open('test_x.pkl', 'rb')
        eval_data = np.asarray(pickle.load(file_open), dtype=np.float32)
        file_open.close()

        file_open = open('test_t.pkl', 'rb')
        eval_labels = np.asarray(pickle.load(file_open), dtype=np.float32)
        file_open.close()

        # Create the Estimator
        facial_classifier = tf.estimator.Estimator(
                model_fn=cnn_model_fn, model_dir="tensorflow/denoise_relu")

        # Set up logging for predictions
        # Log the values in the "Softmax" tensor with label "probabilities"
        tensors_to_log = {"pred": "softmax_tensor",
                          "class": "classes"}
        logging_hook = tf.train.LoggingTensorHook(
                tensors=tensors_to_log, every_n_iter=50)

        # Train the model
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"x": train_data},
                y=train_labels,
                batch_size=10,
                num_epochs=50,
                shuffle=True)
        
        facial_classifier.train(
                input_fn=train_input_fn,
                steps=20000,
                hooks=[logging_hook])

        # Evaluate the model and print results
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"x": eval_data},
                y=eval_labels,
                num_epochs=20,
                shuffle=False)
        eval_results = facial_classifier.evaluate(input_fn=eval_input_fn)
        print(eval_results)

if __name__ == "__main__":
    tf.app.run()