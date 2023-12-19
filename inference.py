# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 12:46:27 2023

@author: shail
"""


import pandas as pd
import numpy as np
import librosa #To deal with Audio files
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder
import pickle
import h5py


# Create a new model instance
model = tf.keras.models.load_model('./models/my_model')
model.load_weights('./checkpoints/my_checkpoint')



spectrogram_shape = (128, 128)

X = []

filepath = input("input file name : ")
normal_sound_sample,sample_rate = librosa.load(filepath)
spectrogram = librosa.amplitude_to_db(np.abs(librosa.stft(normal_sound_sample)), ref=np.max)
spectrogram = np.resize(spectrogram, spectrogram_shape)
X.append(spectrogram)

X = np.array(X)
X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

y_pred = model(X)

with open('labels.pkl', 'rb') as output:
    labelencoder = pickle.load(output)

class_label  = labelencoder.inverse_transform(np.argmax(y_pred,axis=1))

print(class_label[0])
