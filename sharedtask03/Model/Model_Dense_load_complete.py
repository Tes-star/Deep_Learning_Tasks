import keras_cv
from imgaug import augmenters as iaa
import numpy as np
from keras.backend import random_normal
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils.layer_utils import count_params
from sklearn.utils import shuffle
from wandb.integration.keras import WandbCallback

import wandb
from keras.utils import to_categorical
from tensorflow import keras
import tensorflow as tf
import keras
from keras.layers import BatchNormalization, SpatialDropout2D, Activation, Concatenate
from keras.layers import Dropout

from sharedtask03.Model.helper import import_images


def get_model(config):
    def rescale(image):
        image = tf.cast(image, tf.float32)
        image = (image / 255.0)
        return image

    ###################################################################################################################
    dropout = config.dropout_rate

    # loop
    block_len = config.block_len

    cnn_unit = config.cnn_unit

    block_qty = config.block_qty

    unit_list = []
    for i in range(block_qty):
        unit_list.append(cnn_unit)
    wandb.log({'unit_list': unit_list})

    ###################################################################################################################

    input = keras.Input((32, 32, 3))
    x = input
    x = rescale(x)

    # first layer
    x = keras.layers.Conv2D(filters=unit_list[0], kernel_size=3, padding='same', use_bias=False  # ,
                            # kernel_regularizer=l2(wt_decay),
                            # kernel_initializer=(random_normal(stddev=np.sqrt(2.0 / (9 * int(units))))))
                            )(x)
    concat = x

    for i, units in enumerate(unit_list):
        for j in range(block_len):
            x = SpatialDropout2D(dropout)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = keras.layers.Conv2D(filters=units, kernel_size=3, padding='same', use_bias=False  # ,
                                    # kernel_regularizer=l2(wt_decay),
                                    # kernel_initializer=(random_normal(stddev=np.sqrt(2.0 / (9 * int(units))))))
                                    )(x)

            if j > 0 and j != block_len:
                x = Concatenate(axis=-1)([concat, x])
            if j == 0:
                concat = x
        x = keras.layers.MaxPool2D()(x)

    # output
    flat = keras.layers.Flatten()(x)
    # apply dense layer for classification: (None, 16) -> (None, 10)
    output = keras.layers.Dense(units=9, activation='softmax')(flat)

    # build and compile model
    model = keras.Model(input, output)
    model.summary()
    print('weights: ' + str(count_params(model.trainable_weights)))
    wandb.log({'weights': count_params(model.trainable_weights)})
    return model


def train_model():
    print("Run start.")

    wandb.init(dir="C:\\temp")
    config = wandb.config

    if config.optimizer == 'Adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
        # lr=0.001,decay=0, beta_1=0.9, beta_2=0.999, epsilon=1e-08
    if config.optimizer == 'SGD':
        optimizer = tf.keras.optimizers.SGD(learning_rate=config.learning_rate)
    if config.optimizer == 'RMSprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=config.learning_rate)
    if config.optimizer == 'Adadelta':
        optimizer = tf.keras.optimizers.Adadelta(learning_rate=config.learning_rate)
    if config.optimizer == 'Adamax':
        optimizer = tf.keras.optimizers.Adamax(learning_rate=config.learning_rate)
    if config.optimizer == 'Nadam':
        optimizer = tf.keras.optimizers.Nadam(learning_rate=config.learning_rate)
    if config.optimizer == 'Adagrad':
        optimizer = tf.keras.optimizers.Adagrad(learning_rate=config.learning_rate)
    if config.optimizer == 'Ftrl':
        optimizer = tf.keras.optimizers.Ftrl(learning_rate=config.learning_rate)

    # train_ds, test_ds = get_data(config)

    # import data

    y_train, x_train = import_images(path='../data/01_train/train/')
    y_test, x_test = import_images(path='../data/01_train/val/')

    x_train, y_train = shuffle(x_train, y_train, random_state=0)
    x_test, y_test = shuffle(x_test, y_test, random_state=0)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # Augmentation
    wandb.log({'Augmentation': True})

    AUTO = tf.data.AUTOTUNE
    BATCH_SIZE = config.batch_size
    EPOCHS = 1
    IMAGE_SIZE = 32
    rand_aug = iaa.RandAugment(n=config.augmentations_per_image, m=config.magnitude)
    #rand_aug = iaa.RandAugment(n=0, m=0)

    def augment(images):
        # Input to `augment()` is a TensorFlow tensor which
        # is not supported by `imgaug`. This is why we first
        # convert it to its `numpy` variant.
        images = tf.cast(images, tf.uint8)
        return rand_aug(images=images.numpy())

    # create dataholder
    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(BATCH_SIZE * 100)
        .batch(BATCH_SIZE)
        .map(
            lambda x, y: (tf.py_function(augment, [x], [tf.float32])[0], y),
            num_parallel_calls=AUTO,
        )
        .prefetch(AUTO)
    )

    test_ds = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test))
        .batch(BATCH_SIZE)
        .prefetch(AUTO)
    )

    model = get_model(config)

    callbacks = [WandbCallback()]  #

    patience = 5
    wandb.log({'patience': patience})

    early_stopping = EarlyStopping(
        monitor='val_categorical_accuracy',
        min_delta=0.000,  # minimium amount of change to count as an improvement
        patience=patience,  # how many epochs to wait before stopping
        restore_best_weights=False,
    )
    callbacks.append(early_stopping)

    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['categorical_accuracy'],
                  optimizer=optimizer)

    model.fit(train_ds, epochs=1000, shuffle=True,
              validation_data=test_ds,
              callbacks=callbacks
              )
    print("Run ended successful!")
    print()


if __name__ == '__main__':
    # define sweep_id
    sweep_id = 'cr28g6pk'
    # sweep_id = wandb.sweep(sweep=sweep_configuration, project='Abgabe_02', entity="deep_learning_hsa")
    # run the sweep

    wandb.agent(sweep_id, function=train_model, project="Abgabe_03",
                entity="deep_learning_hsa")
