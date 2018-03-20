# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 02:29:42 2018

@author: Jack
"""

import tensorflow as tf
import pickle
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)
#images are 112x112, this should be a flag

def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    #input_layer = tf.reshape(features["x"], [-1, dims, dims, 1])
    #print('input layer',input_layer.shape)
   # flat = tf.reshape(input_layer, -1, dims * dims * 10)
    # Convolutional Layer #1
    # Padding is added to preserve width and height.
    in_layer = tf.reshape(features["x"], [-1, 13225])
    print(in_layer.shape)
    dense1 = tf.layers.dense(
        inputs=in_layer,
        units = 4096,
        activation=tf.nn.relu)
    dense2 = tf.layers.dense(
        inputs=dense1,
        units = 3200,
        activation=tf.nn.relu)
    dense3 = tf.layers.dense(
        inputs=dense2,
        units = 1024,
        activation=tf.nn.relu)
    dense4 = tf.layers.dense(
        inputs=dense3,
        units = 512,
        activation=tf.nn.relu)
    dense5 = tf.layers.dense(
        inputs=dense4,
        units = 512,
        activation=tf.nn.relu)
    dense6 = tf.layers.dense(
        inputs=dense5,
        units = 512,
        activation=tf.nn.relu)
    dense7 = tf.layers.dense(
        inputs=dense6,
        units = 512,
        activation=tf.nn.relu)
    dense8 = tf.layers.dense(
        inputs=dense7,
        units = 1024,
        activation=tf.nn.relu)
    dense9 = tf.layers.dense(
        inputs=dense8,
        units = 3200,
        activation=tf.nn.relu)
    dense10 = tf.layers.dense(
        inputs=dense9,
        units = 4096,
        activation=tf.nn.relu)
    final_dense = tf.layers.dense(
        inputs=dense10,
        units = 13225,
        activation=tf.nn.relu)
    
    print(dense1.shape, final_dense.shape)


    square = tf.reshape(final_dense, [-1, 115, 115])
    # Normal loss function
    labels = tf.identity(labels, name='y')
    loss = tf.reduce_mean(tf.square(square - labels))

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001)
        train_op = optimizer.minimize(loss=loss,
                global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
        labels=labels, predictions=square)}
    return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(unused_argv):
    with tf.device('/gpu:0'):
        # Load training and eval data
        file_open = open('0009/M3/train/full_set.pkl', 'rb')
        train_data = np.asarray(pickle.load(file_open), dtype=np.float32)
        file_open.close()

        file_open = open('0009/M0/train/full_set.pkl', 'rb')
        train_labels =np.asarray(pickle.load(file_open), dtype=np.float32)
        file_open.close()

        file_open = open('0009/M3/test/full_set.pkl', 'rb')
        eval_data = np.asarray(pickle.load(file_open), dtype=np.float32)
        file_open.close()

        file_open = open('0009/M0/test/full_set.pkl', 'rb')
        eval_labels = np.asarray(pickle.load(file_open), dtype=np.float32)
        file_open.close()

        # Create the Estimator
        facial_classifier = tf.estimator.Estimator(
                model_fn=cnn_model_fn, model_dir="tensorflow/denoise_relu")

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