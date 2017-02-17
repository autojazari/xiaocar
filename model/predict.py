import os
import json
import pickle

import cv2

import keras.backend
from keras.models import model_from_json

global model

_location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

model_file = os.path.join(_location__, 'outputs/robot.json')
weights_file = os.path.join(_location__, "weights-improvement-68-0.1653.h5")

with open(model_file, 'r') as jfile:
    model = model_from_json(json.load(jfile))

model.compile("adam", "mse")

model.load_weights(weights_file)

def predict(frame):
    image = cv2.imread(frame, 3)

    img = cv2.resize(image, (64, 48))

    img = img.reshape((1, 48, 64, 3))

    return model.predict(img, batch_size=1)
