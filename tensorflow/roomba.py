#!/usr/bin/env python3.6

import tensorflow as tf
import rospy
from geometry_msgs.msg import Twist
import cv2
import sys
import os
from PIL import Image
import numpy as np

MODEL_PATH = '/home/jetson/Documents/Robotics/Tensorflow'
VIDEO_PATH = '/home/jetson/Documents/Robotics/videos'
model = tf.keras.models.load_model(os.path.join(MODEL_PATH, 'jetson_cnn.h5'))


def predict_image(image):
    '''
    Predicts if there is an intersection ahead. Requires a PIL Image as input
    '''
    img_width = 180
    img_height = 180
    dim = (img_width, img_height)

    image = image.resize(dim)
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    return score


def move_circle():

    pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
    # Create a Twist message and add linear x and angular z values
    move_cmd = Twist()
    move_cmd.linear.x = 0.0
    move_cmd.angular.z = 1.7

    # Save current time and set publish rate at 10 Hz
    now = rospy.Time.now()
    rate = rospy.Rate(10)

    # For the next 1 seconds publish cmd_vel move commands to
    while rospy.Time.now() < now + rospy.Duration.from_sec(1):
        pub.publish(move_cmd)
        rate.sleep()

    # Save current time and set publish rate at 10 Hz
   # For the next 1 seconds publish cmd_vel move commands to
    # Opens video using OpenCV
    cap = cv2.VideoCapture(os.path.join(VIDEO_PATH, 'output.ogv'))

    while cap.isOpened():
        ret, frame = cap.read()

        cv2.imshow('Frame', frame)

        snap = Image.fromarray(frame.astype('uint8'), 'RGB')
        prediction, score = predict_image(snap)
        confidence = 100*np.max(score)
        if (confidence >= 25):
            print(0)  # Lane prediction
        else:
            print(1)  # Intersection prediction
            move_cmd.linear.x = 0.0
            move_cmd.angular.z = 1.7

    # Save current time and set publish rate at 10 Hz
            now = rospy.Time.now()
            rate = rospy.Rate(10)

    # For the next 1 seconds publish cmd_vel move commands to
            while rospy.Time.now() < now + rospy.Duration.from_sec(1):
                pub.publish(move_cmd)
                rate.sleep()

        if cv2.waitKey(10) & 0xFF == ord('q'):
            sys.exit()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        rospy.init_node('roomba_controller', anonymous=True)
        move_circle()
    except rospy.ROSInterruptException:
        pass
