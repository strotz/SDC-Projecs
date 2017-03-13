from sklearn import svm
from scipy.misc import imresize
import glob
import cv2
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg

from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.models import load_model

class TrainingSet:
    def __init__(self, size=32):
        self.size = size

    def LoadImages(self, path):
        images = []
        for file in glob.glob(path, recursive=True):
            image = mpimg.imread(file)
            image = imresize(image, (self.size, self.size))
            images.append(image)
        return np.asarray(images)

    def PrintStats(self, array):
        print(array.shape)
        print(array.dtype)

        print("Mean: ", np.mean(array))
        print("Min: ",  np.min(array))
        print("Max: ", np.max(array))
        print("STD: ", np.std(array))


    def LoadTrainingData(self, test_split=0.05):
        cars_images = self.LoadImages('./data_cars/**/*.png')
        notcars_images = self.LoadImages('./data_nocars/**/*.png')
        print('Cars: {}, No: {} '.format(cars_images.shape[0], notcars_images.shape[0]))

        X = np.concatenate((cars_images, notcars_images), axis=0)
        y = np.hstack((np.ones(cars_images.shape[0]), np.zeros(notcars_images.shape[0]))).flatten()


        rand_state = np.random.randint(0, 100)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_split, random_state=rand_state)


class Detector:
    def __init__(self, size=32):
        self.size = size

    def Build(self):
        size = self.size
        model = Sequential()
        model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(size, size, 3), output_shape=(size, size, 3)))
        model.add(Convolution2D(12, 3, 3, subsample=(2, 2)))
        model.add(Dropout(0.5))
        model.add(Activation('relu'))
        model.add(Convolution2D(24, 3, 3, subsample=(2, 2)))
        model.add(Dropout(0.5))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(80))
        model.add(Dropout(0.5))
        model.add(Activation('relu'))
        model.add(Dense(5))
        model.add(Activation('relu'))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        model.compile('adam', 'binary_crossentropy', ['accuracy']) # rmsprop
        return model

    def Train(self, X, y):
        self.model = self.Build()
        self.history = self.model.fit(X, y, nb_epoch=25, validation_split=0.1, batch_size=128)

    def Test(self, X, y):
        #y_one_hot_test = self.label_binarizer.fit_transform(y)
        metrics = self.model.evaluate(X, y)
        for metric_i in range(len(self.model.metrics_names)):
            metric_name = self.model.metrics_names[metric_i]
            metric_value = metrics[metric_i]
            print('{}: {}'.format(metric_name, metric_value))

    def Detect(self, X):
        return self.model.predict(X, batch_size=128)

    def Save(self, fname):
        self.model.save(fname)

    def Load(self, fname):
        self.model = load_model(fname)
