from keras.callbacks import EarlyStopping
from wandb.integration.keras import WandbCallback

import wandb
from keras.utils import to_categorical
from tensorflow import keras
import tensorflow as tf
import keras
from keras.layers import BatchNormalization
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

    ###################################################################################################################
    dropout = config.dropout_rate

    # loop
    block_len = config.block_len

    cnn_unit = config.cnn_unit

    block_qty = config.block_qty

    unit_list = []
    for i in range(block_qty):
        unit_list.append(2 ** i * cnn_unit)
    wandb.log({'unit_list': unit_list})

    ###################################################################################################################

    input = keras.Input((32, 32, 3))
    x = input
    x = rescale(x)

    for i, units in enumerate(unit_list):
        for j in range(block_len):
            x = SpatialDropout2D(dropout)(x)
            x = BatchNormalization()(x)
            x = keras.layers.Conv2D(filters=units, kernel_size=3, padding='same')(x)
            if j > 0 and j != block_len:
                x = keras.layers.Add()([x, x_last])
            x_last = x
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

    train_ds, test_ds = get_data(config)
    model = get_model(config)
    print('Number of training batches: %d' % tf.data.experimental.cardinality(train_ds).numpy())

    # callbacks

    callbacks = []  #

    #callbacks.append(WandbCallback())
    patience=10
    wandb.log({'patience': patience})

    early_stopping = EarlyStopping(
        monitor='val_categorical_accuracy',
        min_delta=0.000,  # minimium amount of change to count as an improvement
        patience=patience,  # how many epochs to wait before stopping
        restore_best_weights=True,
    )
    callbacks.append(early_stopping)
    wandb.log({'early_stopping': early_stopping})

    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['categorical_accuracy'])

    model.fit(train_ds, epochs=1000, shuffle=True,
              validation_data=test_ds,
              callbacks=callbacks
              )
    print("Run ended successful!")
    print()


if __name__ == '__main__':
    # define sweep_id
    sweep_id = 'hzm5j426'
    # sweep_id = wandb.sweep(sweep=sweep_configuration, project='Abgabe_02', entity="deep_learning_hsa")
    # run the sweep
    wandb.agent(sweep_id, function=train_model, project="Abgabe_03",
                entity="deep_learning_hsa")
