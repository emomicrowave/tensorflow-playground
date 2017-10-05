import numpy as np
import tensorflow as tf

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

LEARNING_RATE = 0.002
EPOCHS = 300

# shortcut function to easily define hidden layers
def hidden_layer(t_input, w_shape):
    h_layer = tf.Variable(tf.random_normal(w_shape, stddev=0.1))
    return tf.nn.sigmoid(tf.matmul(t_input, h_layer))

def load_dataset():      
    data = load_iris()
    x = data.data
    y = data.target.reshape((-1,1))

    # change the dummy variables to one-hot encoding
    enc = OneHotEncoder(sparse=False)
    enc.fit(y)
    return train_test_split(x, enc.transform(y))

def main():
    train_x, test_x, train_y, test_y = load_dataset()

    # layer sizes (features and classes)
    x_size = train_x.shape[1]
    y_size = train_y.shape[1]
    h_size1 = 8
    h_size2 = 16

    print(x_size, y_size)

    # Inputs for feeding into the NN
    X = tf.placeholder(tf.float32, shape=[None, x_size])
    y = tf.placeholder(tf.float32, shape=[None, y_size])

    # Actual model
    h_layer1 = hidden_layer(X, [x_size, h_size1])
    h_layer2 = hidden_layer(h_layer1, [h_size1, h_size2])
    y_hat = hidden_layer(h_layer2, [h_size2, y_size])

    # Back-propagation
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_hat))
    updates = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)

    # Initialize Variables
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    # Train and test
    for epoch in range(EPOCHS):
        #TODO: use a batch to train
        
        # Training step
        for i in range(train_x.shape[0]):
            sess.run(updates, feed_dict={X: train_x[i:i+1], y: train_y[i: i+1]})

        #sess.run(updates, feed_dict={X: train_x, y: train_y})

        correct_prediction = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print("Epoch %d: Accuraccy = %f" % (epoch, sess.run(accuracy, feed_dict={X: test_x, y: test_y})))

if __name__ == "__main__":
    main()
