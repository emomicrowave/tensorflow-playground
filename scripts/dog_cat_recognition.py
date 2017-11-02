import numpy as np
import tensorflow as tf

def model(data, labels, training=True):
    # Input Layer [ features, img_width, img_heigth, channels ] 
    input_layer = tf.reshape(data, [None, 200, 200, 1])

    # Conv Layer 1
    h_conv1 = tf.layers.conv2d(
            inputs = input_layer,
            filters = 32,
            kernel_size = [5,5],
            padding = "same",
            activation = tf.nn.relu )

    # Pooling Layer 1
    h_pool1 = tf.layers.max_pooling2d(
            inputs = h_conv1,
            pool_size = [2,2],
            strides = 2)

    # Conv Layer 2
    h_conv2 = tf.layers.conv2d(
            inputs = h_pool1,
            filters = 64,
            kernel_size = [5,5],
            padding = 'same',
            activation = tf.nn.relu )

    # Pooling Layer 2
    h_pool2 = tf.layer.max_pooling2d(
            inputs = h_conv2,
            pool_size = [2,2],
            strides = 2)

    # Conv Layer 3
    h_conv3 = tf.layers.conv2d(
            inputs = h_pool2,
            filters = 24,
            kernel_size = [5,5],
            padding = 'same',
            activation = tf.nn.relu )

    # Pooling Layer 3
    h_pool3 = tf.layer.max_pooling2d(
            inputs = h_conv3,
            pool_size = [2,2],
            strides = 2)

    # Dense Layer
    h_flat = tf.reshape(h_pool3, [None, 24 * 185 * 185])
    h_dense = tf.layers.dense(inputs=h_flat, units=1024, activation=tf.nn.relu)
    dropout - tf.layers.dropout(inputs=h_dense, rate=0.4 * training)

    # Logits
    logits = tf.layers.dense(inputs=dropout, units=2)

    # TODO: Actually make this work
