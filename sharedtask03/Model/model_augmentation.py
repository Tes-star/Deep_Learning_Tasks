# train with image augmentation
# https://github.com/moritzhambach/Image-Augmentation-in-Keras-CIFAR-10-/blob/master/CNN%20with%20Image%20Augmentation%20(CIFAR10).ipynb
import keras
import keras_cv
from keras.callbacks import EarlyStopping
from wandb.integration.keras import WandbCallback

import wandb
import tensorflow as tf

from sharedtask03.Model.Model_Dense import get_model


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
                                                       batch_size=10000
                                                       )
    # pip install keras-cv
    rand_augment = keras_cv.layers.RandAugment(
        value_range=(0, 255),
        augmentations_per_image=100,
        magnitude=0.3,
        magnitude_stddev=0.2,
        rate=0.5,
    )

    def apply_rand_augment(inputs):
        inputs["images"] = rand_augment(inputs["images"])
        return inputs

    train_ds = train_ds.map(apply_rand_augment, num_parallel_calls=1)

    return train_ds, test_ds



def train_model():
    print("Run start.")

    wandb.init(reinit=True)
    config = wandb.config

    match config.optimizer:
        case 'Adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
            #lr=0.001,decay=0, beta_1=0.9, beta_2=0.999, epsilon=1e-08
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
    #wandb.log({'early_stopping': early_stopping})

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
