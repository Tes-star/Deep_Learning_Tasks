import keras_cv
import numpy as np
from keras.backend import random_normal
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.regularizers import l2
from wandb.integration.keras import WandbCallback

import wandb
from keras.utils import to_categorical
from tensorflow import keras
import tensorflow as tf
import keras
from keras.layers import BatchNormalization, SpatialDropout2D, Activation, Concatenate
from keras.layers import Dropout


# or from tensorflow import keras

# keras.backend.set_image_data_format('channels_last')
def get_data(config):
    path = '../data/01_train/train/'

    train_ds = keras.utils.image_dataset_from_directory(path,
                                                        image_size=(32, 32),
                                                        color_mode='rgb',
                                                        seed=1234,
                                                        labels='inferred',
                                                        label_mode='categorical',
                                                        batch_size=32)

    path = '../data/01_train/val/'
    test_ds = keras.utils.image_dataset_from_directory(path,
                                                       image_size=(32, 32),
                                                       color_mode='rgb',
                                                       seed=1234,
                                                       labels='inferred',
                                                       label_mode='categorical',
                                                       batch_size=10000)
    return train_ds, test_ds


def get_model(config):
    def rescale(image):
        image = tf.cast(image, tf.float32)
        image = (image / 255.0)
        return image

    rand_augment = keras_cv.layers.RandAugment(
        value_range=(0, 255),
        augmentations_per_image=3,
        magnitude=0.3,
        magnitude_stddev=0.2,
        rate=0.5,
    )

    ###################################################################################################################
    dropout = config.dropout_rate

    # loop
    block_len = config.block_len

    cnn_unit = config.cnn_unit

    block_qty = config.block_qty

    unit_list = []
    for i in range(block_qty):
        unit_list.append(1 ** i * cnn_unit)
    wandb.log({'unit_list': unit_list})

    wt_decay = 0.001
    ###################################################################################################################

    input = keras.Input((32, 32, 3))
    x = input
    #x = rand_augment(x)
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
    return model


def train_model():
    print("Run start.")

    wandb.init(reinit=True)
    config = wandb.config

    match config.optimizer:
        case 'Adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
            # lr=0.001,decay=0, beta_1=0.9, beta_2=0.999, epsilon=1e-08
        case 'SGD':
            optimizer = tf.keras.optimizers.SGD(learning_rate=config.learning_rate)
        case 'RMSprop':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=config.learning_rate)
        case 'Adadelta':
            optimizer = tf.keras.optimizers.Adadelta(learning_rate=config.learning_rate)
        case 'Adamax':
            optimizer = tf.keras.optimizers.Adamax(learning_rate=config.learning_rate)
        case 'Nadam':
            optimizer = tf.keras.optimizers.Nadam(learning_rate=config.learning_rate)
        case 'Adagrad':
            optimizer = tf.keras.optimizers.Adagrad(learning_rate=config.learning_rate)
        case 'Ftrl':
            optimizer = tf.keras.optimizers.Ftrl(learning_rate=config.learning_rate)

    train_ds, test_ds = get_data(config)
    model = get_model(config)
    print('Number of training batches: %d' % tf.data.experimental.cardinality(train_ds).numpy())

    # callbacks

    callbacks = []  #

    callbacks.append(WandbCallback())
    patience = 3
    wandb.log({'patience': patience})

    early_stopping = EarlyStopping(
        monitor='val_categorical_accuracy',
        min_delta=0.000,  # minimium amount of change to count as an improvement
        patience=patience,  # how many epochs to wait before stopping
        restore_best_weights=True,
    )
    callbacks.append(early_stopping)
    # wandb.log({'early_stopping': early_stopping})

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
    sweep_id = 'r3qma8ub'
    # sweep_id = wandb.sweep(sweep=sweep_configuration, project='Abgabe_02', entity="deep_learning_hsa")
    # run the sweep
    wandb.agent(sweep_id, function=train_model, project="Abgabe_03",
                entity="deep_learning_hsa")
