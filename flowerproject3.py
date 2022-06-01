import tensorflow as tf
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.losses as losses
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.utils as utils

"""
phys = tf.config.list_physical_devices('GPU')
tf.config.set_logical_device_configuration(
    phys[0],
    [
            tf.config.LogicalDeviceConfiguration(memory_limit=2048),
    ]
)
"""
train = utils.image_dataset_from_directory(
    'flowerclasseddataset',
    label_mode = 'categorical',
    image_size = (226, 226),
    shuffle = True,
    seed = 420,
    validation_split = 0.3,
    subset = 'training',
)



test = utils.image_dataset_from_directory(
    'flowerclasseddataset',
    label_mode = 'categorical',
    image_size = (226, 226),
    shuffle = True,
    seed = 420,
    validation_split = 0.3,
    subset = 'validation',
)

# Data augmentation for more images, same # of images
# dropout (after conv after pooling - 1st half), (any dense layer - not last) --> dropout can be bigger earlier

rotation = models.Sequential([
    layers.RandomFlip("horizontal_and_vertical", input_shape = (226, 226, 3)),
    layers.RandomZoom(.5, .2),
    layers.RandomContrast(0.2)
])

rotated = train.map(lambda x, y: (rotation(x), y))
train = train.concatenate(rotated)

class Net():
    def __init__(self, image_size):
        self.model = models.Sequential()

        # First layer is convolution with:
        # Frame/kernel: 11 x 11, Stride: 5x5, Depth: 8, Input size: 226
        self.model.add(layers.Conv2D(8, 11, strides = 5, input_shape = image_size, activation = "relu"))
        # Output: 44 X 44 X 8
        self.model.add(layers.BatchNormalization())
        # model.add.layer - batch normalization: nothing in parentheses - after relu
        self.model.add(layers.MaxPool2D(pool_size = 2))
        # Output: 22 X 22 X 8
        self.model.add(layers.Conv2D(16, 3, strides = 1, activation = "relu")) # depth - 16, frame - 1/20 of input -- 3x3
        # Output: 19 x 19 x 16
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.MaxPool2D(pool_size = 1))
        # try another conv (3x3 stride 1)
        # try another max pool

        # Output: 19 x 19 x 16
        self.model.add(layers.Flatten())
        # try to prevent jump less than 4x
        # Output: 5776
        self.model.add(layers.Dense(1444, activation = "relu"))
        self.model.add(layers.Dense(360, activation = "relu"))
        self.model.add(layers.Dense(102, activation = "softmax"))
        self.loss = (losses.CategoricalCrossentropy()) # categorical cross entropy
        self.optimizer = (optimizers.SGD(learning_rate = 0.001, momentum = 0.5)) # momentum: less?, optimizer: adam
        self.model.compile(
            loss = self.loss,
            optimizer = self.optimizer,
            metrics = ['accuracy'],
        )
    def __str__(self):
        self.model.summary() # prints
        return ""

net = Net((226, 226, 3))
print(net)

# save model
net.model.fit(
    train,
    batch_size = 32,
    epochs = 100, # run 10,000 epochs overnight
    verbose = 2,
    validation_data = test,
    validation_batch_size = 32,
)

net.model.save('flowermodel')
