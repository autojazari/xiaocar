import os
import glob
import json
import random
import csv

import cv2
import numpy as np

from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.core import Lambda
from keras.callbacks import Callback
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

_index_in_epoch = 0

nb_epoch = 300

batch_size = 32

img_height, img_width = 240, 320

global threshold
threshold = 0.04

class LifecycleCallback(Callback):

    def on_epoch_begin(self, epoch, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):
        global threshold
        threshold = threshold + (i * .05) 
        if threshold > 0.8:
            threshold = 0.2


def shuffle(x, y):
    perm = np.arange(len(x))
    np.random.shuffle(perm)
    x = x[perm]
    y = y[perm]

    return (x, y)


def load_training_and_validation():

    # data = pickle.load(open('robot-balanced-data.p', 'rb'))

    # X_train = data['X_train']
    # y_train = data['Y_train']

    # X_test = data['X_val']
    # y_test = data['Y_val']

    # X_train, Y_train = shuffle(X_train, y_train)
    
    # X_test, Y_test = shuffle(X_test, y_test)

    filenames = glob.glob("../data/*.jpg")
    labels = []
    images = []
    for filename in filenames:
        # 148701337247041-0.773193359375-0.773193359375
        filename = filename.replace('.jpg', '')
        parts = filename.split('-')
        images.append(filename)
        labels.append([float(parts[1]), float(parts[2])])



    X_train, X_val, y_train, y_val = train_test_split(np.array(images), np.array(labels), test_size=0.15, random_state=42)

    return X_train, X_val, y_train, y_val 

def affine_transform(img, controls):

    right = True if np.random.uniform(-1, 1) > 0 else False
    pixels = int(np.random.uniform(10, 50))

    cols, rows, ch = img.shape

    pts_a = [75, 75]
    pts_b = [150, 75]
    pts_c = [50, 250]

    pts = [pts_a, pts_b, pts_c]

    pts1 = np.float32(pts)

    if right:
        pts_a = [pts_a[0]+pixels, 75]
        pts_b = [pts_b[0]+pixels, 75]
        pts_c = [pts_c[0]+pixels, 250]
        # angle -= (.002 * pixels)
        controls = [angle - (.002 * pixels) for angle in controls]
        
    else:
        pts_a = [pts_a[0]-pixels, 75]
        pts_b = [pts_b[0]-pixels, 75]
        pts_c = [pts_c[0]-pixels, 250]
        # angle += (.002 * pixels)
        controls = [angle + (.002 * pixels) for angle in controls]

    pts2 = np.float32([pts_a, pts_b, pts_c])

    M = cv2.getAffineTransform(pts1, pts2)

    dst = cv2.warpAffine(img, M, (rows, cols))

    return dst.reshape((cols, rows, ch)), controls

def next_batch(data, labels, batch_size):
    """Return the next `batch_size` examples from this data set."""
    global _index_in_epoch
    start = _index_in_epoch
    _index_in_epoch += batch_size
    _num_examples = len(data)
    if _index_in_epoch > _num_examples:
        # Shuffle the data
        data, labels = shuffle(data, labels)
        # Start next epoch
        start = 0
        _index_in_epoch = batch_size
        # assert batch_size <= _num_examples
    end = _index_in_epoch
    return data[start:end], labels[start:end]


def transform_generator(x, y, batch_size=32, is_validation=False):
    while True:
        images, labels = list(), list()

        _images, _labels = next_batch(x, y, batch_size)
        current = os.path.dirname(os.path.realpath(__file__))

        # read in images as grayscale
        # affine transform (right and left)
        # to add additional angles
        for i in range(len(_images)):
            original = cv2.imread(_images[i]+'.jpg', 1)
           
            if original is None:
                continue

            img = cv2.resize(original, (64, 48))
            
            controls = list(_labels[i])

            if controls[0] == 0. or controls[1] == 0.:
                continue

            images.append(img)
            labels.append(controls)

            if is_validation: continue

            # flip vertically
            image = cv2.flip(img, 1)
            controls = [c * -1 for c in controls]

            images.append(image)
            labels.append(controls)

            # image, controls = affine_transform(original, controls)
                
            # resized = cv2.resize(image, (64, 48))
            
            # images.append(resized)
                
            # labels.append(controls)

            # angle = abs(controls[1] - controls[0])
            # print("angle", angle, controls)

            # if threshold > angle:
            #     pixels = 15
            #     adjust = .001 * pixels * threshold
            
            #     image, controls = affine_transform(img, pixels, controls, adjust, right=True)
                
            #     resized = resize_image(image)
            #     images.append(resized)
            #     labels.append(controls)
            #     # print(adjust, controls)

            #     pixels = 20
            #     adjust = .001 * pixels * threshold
            
            #     image, controls = affine_transform(img, pixels, controls, adjust, right=True)
            #     resized = resize_image(image)
            #     images.append(resized)
            #     labels.append(controls)
                # print(adjust, controls)

        X = np.array(images, dtype=np.float64)

        Y = np.array(labels, dtype=np.float64)

        yield (X, Y)

def gen_model():
    ch, row, col = 3, 48, 64  # camera format

    model = Sequential()

    model.add(Lambda(lambda x: x/127.5 - 1.,
            input_shape=(row, col, ch),
            output_shape=(row, col, ch)))

    # trim the image
    model.add(Lambda(lambda x: x[:,20::,:]))

    # model.add(Lambda(lambda image: K.resize_images(image, 240./64, 340./64, 'tf')))

    # (((64 - 3) + 0) / 1.) + 1
    model.add(Convolution2D(32, 3, 3, subsample=(1, 1)))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # (((62 - 3) + 0) / 1.) + 1
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1)))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # (((60 - 5) + 4) / 1.) + 1
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1)))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(2))

    adam = Adam(lr=0.0001)

    model.compile(loss='mean_squared_error', optimizer=adam)

    return model

def main():

    X_train, X_val, y_train, y_val = load_training_and_validation()

    assert len(X_train) == len(y_train), 'unbalanced training data'
    assert len(X_val) == len(y_val), 'unbalanced validation data'

    print(len(X_train), "training images and ", len(X_val), "validation images")

    model = gen_model()

    filepath = "weights-improvement-{epoch:02d}-{val_loss:.4f}.h5"
    checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=True)

    model.fit_generator(
        transform_generator(X_train, y_train),
        samples_per_epoch=len(X_train),
        nb_epoch=nb_epoch,
        validation_data=transform_generator(X_val, y_val, is_validation=True),
        nb_val_samples=(len(X_val)),
        callbacks=[checkpoint])

    print("Saving model weights and configuration file.")

    if not os.path.exists("./outputs"):
        os.makedirs("./outputs")

    model.save_weights("./outputs/robot.hd5", True)
    with open('./outputs/robot.json', 'w') as outfile:
        json.dump(model.to_json(), outfile)


if __name__ == '__main__':
    main()
