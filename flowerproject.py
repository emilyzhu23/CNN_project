import tensorflow as tf
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.losses as losses
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.utils as utils

train = utils.image_dataset_from_directory(
    'flowerclasseddataset',
    label_mode = 'categorical',
    image_size = (224, 224),
    shuffle = True,
    seed = 420,
    validation_split = 0.3,
    subset = 'training',
)

test = utils.image_dataset_from_directory(
    'flowerclasseddataset',
    label_mode = 'categorical',
    image_size = (224, 224),
    shuffle = True,
    seed = 420,
    validation_split = 0.3,
    subset = 'validation',
)

class Net():
    def __init__(self, image_size):
        self.model = models.Sequential()
        # First layer is convolution with:
        # Frame/kernel: 11 x 11 (224/20), Stride: 2 x 2, Depth: 8, Input size: 224
        self.model.add(layers.Conv2D(8, 11, strides = 2, input_shape = image_size, activation = "relu"))
        # Output: 214 x 214 x 8
        # Input: 214 x 214 x 8
        self.model.add(layers.MaxPool2D(pool_size = 2))
        # Output: 107 x 107 x 8
        self.model.add(layers.Conv2D(16, 5, activation = "relu")) # depth - 16, frame - 1/20 of input
        # Output: 92 x 92 x 16
        self.model.add(layers.MaxPool2D(pool_size = 2))
        # Output: 46 x 46 x 16
        self.model.add(layers.Flatten())
        # Output: 33,856
        self.model.add(layers.Dense(1024, activation = "relu"))
        self.model.add(layers.Dense(256, activation = "relu"))
        self.model.add(layers.Dense(64, activation = "relu"))
        self.model.add(layers.Dense(5, activation = "softmax"))
        self.loss = (losses.MeanSquaredError())
        self.optimizer = (optimizers.SGD(learning_rate = 0.0001))
        self.model.compile(
            loss = self.loss,
            optimizer = self.optimizer,
            metrics = ['accuracy'],
        )
    def __str__(self):
        self.model.summary() # prints
        return ""

net = Net((224, 224, 3))
print(net)

net.model.fit(
    train,
    batch_size = 32,
    epochs = 100,
    verbose = 2,
    validation_data = test,
    validation_batch_size = 32,
)