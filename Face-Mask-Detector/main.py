import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow import keras
import tensorflow as tf
from PIL import Image
import pickle

# Data Cleaning
print('Cleaning...')
without_mask = os.listdir('data/without_mask')
with_mask = os.listdir('data/with_mask')


def img2array(path):
    img = Image.open(path)
    img = img.convert(('RGB'))
    img = img.resize((128,128))
    img_scaled = np.array(img) / 255
    return np.array(img_scaled)


data = []

for path in with_mask:
    data.append(img2array('data/with_mask/'+path))

for path in without_mask:
    data.append(img2array('data/without_mask/'+path))

labels = [1] * len(with_mask)
labels += [0] * len(without_mask)

X = np.array(data)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# training
print('Training...')
num_of_classes = 2

model = keras.Sequential()

model.add(keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(keras.layers.Conv2D(64, (3,3), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Dense(64, activation=('relu')))
model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Dense(num_of_classes, activation='sigmoid'))

# compiling
print('Compiling...')
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train, y_train, epochs=5)

#saving model
print('Saving...')
pickle.dump(model, open('model.pkl','wb'))

print('---------------completed--------------')