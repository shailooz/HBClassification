# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 11:42:17 2023

@author: shail
"""

import os
import glob
import fnmatch
import pandas as pd
import numpy as np
import librosa #To deal with Audio files
import librosa.display
import matplotlib.pyplot as plt
import IPython.display as ipd
import math
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder
import pickle
import h5py

data_path = "./data/"
spectrogram_shape = (128, 128)

metdatafile =  pd.read_csv("./data/set_a.csv")
metdatafile.label.fillna('others', inplace=True)
print(metdatafile)

X = []
y = []

for row in metdatafile.iloc[:, 1:3].values:
    normal_sound_sample,sample_rate = librosa.load(data_path+row[0])
    spectrogram = librosa.amplitude_to_db(np.abs(librosa.stft(normal_sound_sample)), ref=np.max)
    spectrogram = np.resize(spectrogram, spectrogram_shape)
    X.append(spectrogram)
    y.append(row[1])
    
X = np.array(X)
X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)


labels = np.unique(y)
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)
y_encoded = tf.keras.utils.to_categorical(y)
    
model = tf.keras.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(X.shape[1], X.shape[2], 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(len(labels), activation='softmax'))

model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X,y_encoded,epochs=20, batch_size=16)

# Save the weights
model.save('./models/my_model')
model.save_weights('./checkpoints/my_checkpoint')
with open('labels.pkl', 'wb') as output:
    pickle.dump(labelencoder, output)


