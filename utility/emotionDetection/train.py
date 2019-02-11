from __future__ import division, absolute_import
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected, flatten
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.merge_ops import merge
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from os.path import isfile, join
import random
import sys
import tensorflow as tf
import os
import glob
import cv2
import sys

from model import EMR

print("Start")

# epoch =
# epoch = int(input('input epoch : '))

# prevents appearance of tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# prevents opencl usage and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)

target_classes = ['angry', 'happy', 'neutral', 'sad', 'scared']


def make_sets():
    training_data = []
    training_labels = []

    testing_data = []
    testing_labels = []

    for emotion in target_classes:
        training, testing = get_files(emotion)
        for item in training:
            image = cv2.imread(item)  # open image
            newimg = format_image(image)
            # output = np.array([128, 128, 1])

            if(len(newimg) == 1):
                continue
#             np.concatenate((training_data, newimg))
            training_data.append(newimg)
            arr = target_classes.index(emotion)
#             training_labels.append(target_classes.index(emotion))
            if(arr == 0):
                training_labels.append([1, 0, 0, 0, 0])
            elif(arr == 1):
                training_labels.append([0, 1, 0, 0, 0])
            elif(arr == 2):
                training_labels.append([0, 0, 1, 0, 0])
            elif(arr == 3):
                training_labels.append([0, 0, 0, 1, 0])
            elif(arr == 4):
                training_labels.append([0, 0, 0, 0, 1])

#             training_labels.append(arr)

        for item in testing:
            image = cv2.imread(item)  # open image
            newimg = format_image(image)
            output = np.array([128, 128, 1])

            if(len(newimg) == 1):
                continue
#             np.concatenate((training_data, newimg))
            testing_data.append(newimg)
            arr = target_classes.index(emotion)
#             training_labels.append(target_classes.index(emotion))
            if(arr == 0):
                testing_labels.append([1, 0, 0, 0, 0])
            elif(arr == 1):
                testing_labels.append([0, 1, 0, 0, 0])
            elif(arr == 2):
                testing_labels.append([0, 0, 1, 0, 0])
            elif(arr == 3):
                testing_labels.append([0, 0, 0, 1, 0])
            elif(arr == 4):
                testing_labels.append([0, 0, 0, 0, 1])

#             training_labels.append(arr)

    return training_data, training_labels, testing_data, testing_labels


def get_files(emotion):
    print("./utility/emotionDetection/images/%s/*" % emotion)
    files = glob.glob("./utility/emotionDetection/images/%s/*" % emotion)
    print(len(files))

    random.shuffle(files)

    training = files[:int(len(files)*0.8)]
    testing = files[:int(len(files)*0.2)]

    return training, testing


def format_image(image):
    """
    Function to format frame
    """
    if len(image.shape) > 2 and image.shape[2] == 3:
            # determine whether the image is color
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        # Image read from buffer
        image = cv2.imdecode(image, cv2.CV_LOAD_IMAGE_GRAYSCALE)

    cascade = cv2.CascadeClassifier(
        "./utility/emotionDetection/haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(image, scaleFactor=1.3, minNeighbors=5)

    if not len(faces) > 0:
        return [0]

    # initialize the first face as having maximum area, then find the one with max_area
    max_area_face = faces[0]
    for face in faces:
        if face[2] * face[3] > max_area_face[2] * max_area_face[3]:
            max_area_face = face
    face = max_area_face

    # extract ROI of face
    image = image[face[1]:(face[1] + face[2]), face[0]:(face[0] + face[3])]

    try:
        # resize the image so that it can be passed to the neural network
        image = cv2.resize(
            image, (48, 48), interpolation=cv2.INTER_CUBIC) / 255.
    except Exception:
        print("----->Problem during resize")
        return None

#     return image
    output = np.array(image)
    output = np.reshape(output, (48, 48, 1))

    return output


# # Create Training set and Testing Set
training_data, training_labels, testing_data, testing_labels = make_sets()

try:
    epoch = int(sys.argv[1])
except:
    epoch = 50

# # Convert to numpy array
training_data = np.array(training_data)
training_labels = np.array(training_labels)

testing_data = np.array(testing_data)
testing_labels = np.array(testing_labels)

network = EMR()
network.build_network()

network.train(training_data, training_labels,
              testing_data, testing_labels, epoch)

print("Finish")

# # Create NN Model
# network = tflearn.input_data(shape=[None, 48, 48, 1])
# print("Input data     ", network.shape[1:])


# network = conv_2d(network, 64, 5, activation='relu')
# print("Conv1          ", network.shape[1:])

# network = max_pool_2d(network, 3, strides=2)
# print("Maxpool1       ", network.shape[1:])


# network = conv_2d(network, 64, 5, activation='relu')
# print("Conv2          ", network.shape[1:])

# network = max_pool_2d(network, 3, strides=2)
# print("Maxpool2       ", network.shape[1:])


# network = conv_2d(network, 128, 4, activation='relu')
# print("Conv3          ", network.shape[1:])

# network = fully_connected(network, 3072, activation='relu')
# print("Fully connected", network.shape[1:])

# network = fully_connected(network, len(target_classes), activation='softmax')
# print("Output         ", network.shape[1:])
# # Generates a TrainOp which contains the information about optimization process - optimizer, loss function, etc

# # # Generates a TrainOp which contains the information about optimization process - optimizer, loss function, etc


# network = regression(network, optimizer='momentum',
#                      metric='accuracy', loss='categorical_crossentropy')

# # Creates a model instance.
# model = tflearn.DNN(network)

# # Create Training set and Testing Set
# training_data, training_labels, testing_data, testing_labels = make_sets()

# # Convert to numpy array
# training_data = np.array(training_data)
# training_labels = np.array(training_labels)

# testing_data = np.array(testing_data)
# testing_labels = np.array(testing_labels)

# # Load Model
# if isfile('model.tflearn.meta'):
#     model.load('model.tflearn')
# else:
#     print('can not see last model')
# # Training
# model.fit(training_data, training_labels, n_epoch=epoch,
#           validation_set=(testing_data, testing_labels), snapshot_step=200)

# # Save Model
# model.save('model_' + str(epoch)+'.tflearn')


# # Evaluate Model
# print('accuracy : ', model.evaluate(testing_data, testing_labels))
# # model.evaluate(testing_data, testing_labels)
