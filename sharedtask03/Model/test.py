import pandas as pd
from keras.utils import to_categorical
from tensorflow import keras
import tensorflow as tf
import keras
from keras.layers import BatchNormalization

path = '../data/01_train/train/'
train_ds = keras.utils.image_dataset_from_directory(path,
                                                    image_size=(32, 32),
                                                    color_mode='rgb',
                                                    labels='inferred',
                                                    label_mode='categorical')

path = '../data/01_train/val/'
test_ds = keras.utils.image_dataset_from_directory(path,
                                                   image_size=(32, 32),
                                                   color_mode='rgb',
                                                   labels='inferred',
                                                   label_mode='categorical')


def get_model():
    # input in the shape of a single digit
    input = keras.Input((32, 32, 3))
    # reshape to match requirements of convolutional layer: (None, 28, 28) -> (None, 28, 28, 1)
    # expanded_input=tf.expand_dims(input, axis=-1)

    # Block 1: (None, 28, 28, 1) -> (None, 14, 14, 4)
    conv_1 = keras.layers.Conv2D(filters=4, kernel_size=3, padding='same')(input)
    conv_2 = keras.layers.Conv2D(filters=4, kernel_size=3, padding='same')(conv_1)
    max_pool_1 = keras.layers.MaxPool2D()(conv_2)

    # Block 2: (None, 14, 14, 4) -> (None, 7, 7, 4)
    conv_3 = keras.layers.Conv2D(filters=4, kernel_size=3, padding='same')(max_pool_1)
    conv_4 = keras.layers.Conv2D(filters=4, kernel_size=3, padding='same')(conv_3)
    # Residual connection
    res_1 = keras.layers.Add()([max_pool_1, conv_4])
    max_pool_2 = keras.layers.MaxPool2D()(res_1)

    # Block 3: (None, 7, 7, 4) -> (None, 7, 7, 4)
    conv_5 = keras.layers.Conv2D(filters=4, kernel_size=3, padding='same')(max_pool_2)
    conv_6 = keras.layers.Conv2D(filters=4, kernel_size=3, padding='same')(conv_5)
    # Residual connection
    res_2 = keras.layers.Add()([max_pool_2, conv_6])

    # (None, 7, 7, 4) -> (None, 5, 5, 8)
    conv_7 = keras.layers.Conv2D(filters=4, kernel_size=3, )(res_2)
    # (None, 5, 5, 8) -> (None, 3, 3, 16)
    conv_8 = keras.layers.Conv2D(filters=4, kernel_size=3, )(conv_7)
    # (None, 3, 3, 16) -> (None, 1, 1, 16)
    conv_9 = keras.layers.Conv2D(filters=4, kernel_size=3, )(conv_8)

    # flatten input into a single dimension: (None, 1, 1, 16) -> (None, 16)
    flat = keras.layers.Flatten()(conv_9)

    # apply dense layer for classification: (None, 16) -> (None, 10)
    output = keras.layers.Dense(units=9, activation='softmax')(flat)
    # build and compile model
    model = keras.Model(input, output)
    return model


model = get_model()
# model.summary()
print('Number of training batches: %d' % tf.data.experimental.cardinality(train_ds).numpy())

model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['categorical_accuracy'])

print(tf.config.list_physical_devices('GPU'))
model.fit(train_ds, epochs=100, validation_data=test_ds)
