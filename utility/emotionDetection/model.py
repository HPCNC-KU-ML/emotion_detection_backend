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


# prevents appearance of tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("Start")


class EMR:
    def __init__(self):
        self.target_classes = ['angry', 'happy', 'neutral', 'sad', 'scared']

    def build_network(self):
        """
        Build the convnet.
        Input is 48x48
        3072 nodes in fully connected layer
        """
        self.network = input_data(shape=[None, 48, 48, 1])
        print("Input data     ", self.network.shape[1:])

        self.network = conv_2d(self.network, 64, 5, activation='relu')
        print("Conv1          ", self.network.shape[1:])

        self.network = max_pool_2d(self.network, 3, strides=2)
        print("Maxpool1       ", self.network.shape[1:])

        self.network = conv_2d(self.network, 64, 5, activation='relu')
        print("Conv2          ", self.network.shape[1:])

        self.network = max_pool_2d(self.network, 3, strides=2)
        print("Maxpool2       ", self.network.shape[1:])

        self.network = conv_2d(self.network, 128, 4, activation='relu')
        print("Conv3          ", self.network.shape[1:])

        self.network = fully_connected(self.network, 3072, activation='relu')
        print("Fully connected", self.network.shape[1:])

        self.network = fully_connected(self.network, len(
            self.target_classes), activation='softmax')
        print("Output         ", self.network.shape[1:])
        # Generates a TrainOp which contains the information about optimization process - optimizer, loss function, etc

        self.network = regression(
            self.network, optimizer='momentum', metric='accuracy', loss='categorical_crossentropy')

        # Creates a model instance.
        self.model = tflearn.DNN(
            self.network, checkpoint_path='./utility/emotionDetection/model_1_atul', max_checkpoints=1, tensorboard_verbose=2)

        # Loads the model weights from the checkpoint
        self.load_model()

    def predict(self, image):
        """
        Image is resized to 48x48, and predictions are returned.
        """
        if image is None:
            return None
        return self.model.predict(image)

    def load_model(self):
        if isfile('./utility/emotionDetection/model_50.tflearn.meta'):
            self.model.load('./utility/emotionDetection/model_50.tflearn')
        elif isfile('./utility/emotionDetection/model_100.tflearn.meta'):
            self.model.load('./utility/emotionDetection/model_100.tflearn')
        elif isfile('./utility/emotionDetection/model.tflearn.meta'):
            self.model.load('./utility/emotionDetection/model.tflearn')
        else:
            print("---> Couldn't find model")
            # change to argv

    def train(self, training_data, training_labels, testing_data, testing_labels, epoch):
        self.load_model()

        # epoch = int(input('input epoch : '))
        self.model.fit(training_data, training_labels, n_epoch=epoch,
                       validation_set=(testing_data, testing_labels), snapshot_step=200)

        # Save Model
        self.model.save('./utility/emotionDetection/model' + '.tflearn')

        # Evaluate Model
        print('accuracy : ', self.model.evaluate(testing_data, testing_labels))
        # model.evaluate(testing_data, testing_labels)


if __name__ == "__main__":
    print("\n------------Emotion Detection Program------------\n")
    # network = EMR()
    import videofile
    # if(sys.argv[1] == 'videofile'):
    #     import videofile
    # elif(sys.argv[1] == 'jsonfile'):
    #     import jsonfile
