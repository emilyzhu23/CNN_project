import tensorflow as tf
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.losses as losses
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.utils as utils

import numpy as np
net = tf.keras.models.load_model('flowermodel')
img = tf.keras.utils.load_img(
    "testimg.jpg", target_size=(226, 226)
)

img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = net.predict(img_array)
top_k_values, top_k_indices = tf.nn.top_k(predictions, k=5)
print(top_k_values)
pritn(top_k_indices)