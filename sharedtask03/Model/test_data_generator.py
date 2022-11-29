import keras
import keras_cv
import tensorflow as tf
from cutmix_keras import CutMixImageDataGenerator  # Import CutMix
from keras.preprocessing.image import ImageDataGenerator

path = '../data/01_train/train/'

rand_augment = keras_cv.layers.RandAugment(
    value_range=(0, 255),
    augmentations_per_image=3,
    magnitude=0.3,
    magnitude_stddev=0.2,
    rate=0.5,
)
train_generator1 = keras.utils.image_dataset_from_directory(path,
                                                            image_size=(32, 32),
                                                            color_mode='rgb',
                                                            seed=1234,
                                                            labels='inferred',
                                                            label_mode='categorical',
                                                            batch_size=32,
                                                            samples=32)

train_generator2 = keras.utils.image_dataset_from_directory(path,
                                                            image_size=(32, 32),
                                                            color_mode='rgb',
                                                            seed=1234,
                                                            labels='inferred',
                                                            label_mode='categorical',
                                                            batch_size=32,
                                                            samples=32
                                                            )
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
)

# CutMixImageDataGenerator
train_generator = CutMixImageDataGenerator(
    generator1=train_generator1,
    generator2=train_generator2,
    img_size=(32, 32),
    batch_size=32,
)
