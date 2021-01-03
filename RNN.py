# Import Libraries
import tensorflow as tf
import numpy as np
from tensorflow import keras

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

model.compile(optimizer='sgd', loss='mean_squared_error')

x = np.array([2, 3, 4], dtype=float)
y = np.array([10, 15, 20], dtype=float)

model.fit(x, y, epochs=1000)

# Replacement of [2, 3, 4] and will give a prediction of the "y" value
# Note the number [5] can be replaces with any number you desire 
results = model.predict([5])

print(results)
