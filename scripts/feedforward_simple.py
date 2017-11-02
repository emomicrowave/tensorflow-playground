import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Datensatz
data = load_iris()
features = data.data
labels = data.target.reshape((-1, 1))

# OneHotEncoding
enc = OneHotEncoder(sparse=False)
enc.fit_transform(labels)

# Trainings- und Testdaten erzeugen
train_x, test_x, train_y, test_y = train_test_split(
                                features, 
                                enc.transform(labels) )

x_size = train_x.shape[1]
y_size = train_y.shape[1]

# Platzhalter für Input
X = tf.placeholder(tf.float32, shape=[None, x_size])
Y = tf.placeholder(tf.float32, shape=[None, y_size])

# Um dieses Modell zu benutzen, Einführungsstriche entfernen
"""
# NN Modell mit TF Core
def hidden_layer(t_input, w_shape, activation=tf.nn.sigmoid):
    weights = tf.Variable(tf.random_normal(w_shape))
    biases = tf.Variable(tf.random_normal([1, w_shape[1]]))

    return activation(tf.add(tf.matmul(t_input, weights), biases))
"""

h_layer1 = hidden_layer(X, [x_size, 128])
h_layer2 = hidden_layer(h_layer1, [128, 128])
y_hat = hidden_layer(h_layer2, [128, y_size], tf.nn.softmax)

# MM Modell mit tf.layers
h_layer1 = tf.layers.dense(X, 128, activation=tf.nn.sigmoid)
h_layer2 = tf.layers.dense(h_layer1, 128, activation=tf.nn.sigmoid)
y_hat = tf.layers.dense(h_layer2, y_size, activation=tf.nn.softmax)

# Kosten- und Optimierungsfunktion
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=Y, logits=y_hat))

train_step = tf.train.GradientDescentOptimizer(0.005).minimize(loss)

# Unbekannten initialisieren und Session erstellen
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Training Loop
for epoch in range(300):
    for i in range(train_x.shape[0]):
        sess.run(train_step, feed_dict={X: train_x[i:i+1], Y: train_y[i: i+1]})                                                 
        
    if (epoch % 10 == 0):
        correct_prediction = tf.equal(tf.argmax(y_hat, 1), tf.argmax(Y, 1))                                                      
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))                                                       
        print("Epoch %d: Accuraccy = %f, Loss = %f" % (                                                                          
                          epoch,                                                                                                   
                          sess.run(accuracy, feed_dict={X: test_x, Y: test_y}),                                                    
                          sess.run(loss, feed_dict={X: train_x, Y: train_y})))
    
