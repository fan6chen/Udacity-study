import os
import csv
import cv2
import numpy as np
import random

# read data collected into samples
samples = []
with open('/home/clement/Desktop/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

# shuffle and split samples with test_size = 0.2
from sklearn.model_selection import train_test_split
random.shuffle(samples)
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import sklearn

# define generator; data augmentation
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # loop forever so the generator never terminators
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                center_image = cv2.imread(batch_sample[0])
                left_image = cv2.imread(batch_sample[1])
                right_image = cv2.imread(batch_sample[2])

                correction = 0.2
                center_angle = float(batch_sample[3])                
                left_angle = center_angle + correction
                right_angle = center_angle - correction

                images.append(center_image)
                images.append(cv2.flip(center_image, 1))
                images.append(left_image)
                images.append(right_image)
                angles.append(center_angle)
                angles.append(-center_angle)
                angles.append(left_angle)
                angles.append(right_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Activation, Dropout, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt

# data preprocessing: normalization and cropping
model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5,input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((50, 20), (0, 0))))
# NVIDIA net architecture
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# Use Adam optimizer
# 5 Epochs
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=5, pickle_safe=True)

model.save('model.h5')

exit()

