import numpy as np
import tensorflow as tf

# Do some import hackery
models = tf.contrib.keras.models
layers = tf.contrib.keras.layers

# Generate Model
model = models.Sequential()
model.add( layers.Dense( 32, activation='relu', input_dim=100))
model.add( layers.Dense( 1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# Generate dummy data
data = np.random.random((1000, 100))
labels = np.random.randint(2, size=(1000,1))

# Train model
model.fit(data, labels, epochs=200, batch_size=32)
