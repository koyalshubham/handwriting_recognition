import numpy as np
import os
import cv2
from tqdm import tqdm
import pandas as pd
import csv
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU 
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras.models import load_model

with open('X50px.csv', 'r') as infile:
    reader = csv.reader(infile)
    lines = list(reader)

X = []
for row in lines:
    new_row = [float(i) for i in row]
    X.append(new_row)
    
X = np.asarray(X)

with open('y50px.csv', 'r') as infile:
    reader = csv.reader(infile)
    lines = list(reader)
    
y = []
for row in lines:
    new_row = [float(i) for i in row]
    y.append(new_row)
    
y = np.asarray(y)

X_train = X.reshape(X.shape[0], 50, 50, 1)
X_train = X_train.astype('float32')
X_train/=255

number_of_classes = 27
Y_train = np_utils.to_categorical(y, number_of_classes)

# Miniaturized VGG-16 model

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(50,50,1), activation='relu', padding='same'))
#model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(1, 1)))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
#model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(1, 1)))

model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
#model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(1, 1)))

model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
#model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(1, 1)))

model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
#model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(1, 1)))

model.add(Flatten())

model.add(Dense(512, activation='relu', kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu', kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
model.add(Dropout(0.5))
model.add(Dense(27, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01,momentum=0.9), metrics=['accuracy'])

#es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=7)

model.summary()

history = model.fit(X_train, Y_train, batch_size=128, epochs=50, verbose=1)

model.save('hwr50px.h5')
