import os

import cv2
import numpy as np


def import_images(path):
    folders = os.listdir(path)

    image_count = sum([len(files) for r, d, files in os.walk(path)])
    x = np.empty((image_count, 32, 32, 3))
    y = np.empty((image_count, 1))

    # Liste aller Dateien in annotation_folder erstellen
    folders = os.listdir(path)

    # Aus Liste files .hdr Dateien löschen

    i = 0
    label = 0
    for folder in folders:
        images = os.listdir(path + '/' + folder)

        for image in images:
            image_path = path + '/' + folder + '/' + image
            x[i] = cv2.imread(image_path)
            y[i] = label
            i = i + 1
        label = label + 1
        # print(i)
    return y, x


def import_images_abgabe(path):
    folders = os.listdir(path)

    image_count = sum([len(files) for r, d, files in os.walk(path)])
    x = np.empty((image_count, 32, 32, 3))
    y = np.empty((image_count, 1))

    # Liste aller Dateien in annotation_folder erstellen
    folders = os.listdir(path)

    # Aus Liste files .hdr Dateien löschen

    i = 0

    images = os.listdir(path)

    for image in images:
        image_path = path + '/' + image
        x[i] = cv2.imread(image_path)
        i = i + 1

        # print(i)
    return x


if __name__ == '__main__':
    path = '../data/01_train/train/'
    # for folder in folders:
    # print(folder)
    y, x = import_images(path)
    print(x.shape)
    print(y.shape)

    path = '../data/01_train/test/'
    x_abgabe = import_images_abgabe(path)
    print(x_abgabe.shape)
