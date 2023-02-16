#!/usr/bin/env python
# coding: utf-8
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt
import os

VIDEO_PATH = "S:/selfdrivingcar/videos/output4.ogv"
MODEL_PATH = "S:/selfdrivingcar/model/ART_Robot.h5"

model = tf.keras.models.load_model(MODEL_PATH)
class_names = ['intersection', 'lane']

img_height = 180
img_width = 180

def predict_image(image):
    '''
    Predicts if there is an intersection ahead. Requires a PIL Image as input
    '''
    img_width = 180
    img_height = 180
    dim = (img_width, img_height)

    image = image.resize(dim)
    img_array = tf.keras.utils.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    return score

dim = (img_height, img_width)
cap = cv2.VideoCapture(VIDEO_PATH)

while cap.isOpened():
    ret, frame = cap.read()

    cv2.imshow('Frame', frame)

    snap = Image.fromarray(frame.astype('uint8'), 'RGB')
    prediction, score = predict_image(snap)
    confidence = 100 * np.max(score)
    if (confidence <= 25):
        print(0)
    else:
        print(1)
    #     print(class_names[np.argmax(score)] + "-" + str(100*np.max(score)))

    print(confidence)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

