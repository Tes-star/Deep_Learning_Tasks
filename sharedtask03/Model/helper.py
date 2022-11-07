import os

import cv2
import numpy as np


def import_images(path):

    image_count = sum([len(files) for r, d, files in os.walk(path)])

    x=np.empty((image_count,32, 32,3))
    y = np.empty((image_count, 1))

    # Liste aller Dateien in annotation_folder erstellen
    folders = os.listdir(path)

    # Aus Liste files .hdr Dateien löschen
    i=0

    for image in folders:
        image_path=path+'/'+image
        x[i] =cv2.imread(image_path)
        y[i]=i
        i=i+1
        #print(i)
    return y,x


if __name__ == '__main__':
    path = '../data/01_train/train/'
    folders = os.listdir(path)
    for folder in folders:
        print(folder)
        y,x = import_images(path+folder+'/')
        print(x.shape)

