import os
import numpy
import pandas
import keras
import sklearn


def train_val_data_split(pneumonia_list, normal_list, validation_split=0.1):
    n_pneumonia = len(pneumonia_list)
    n_normal = len(normal_list)

    shuffle_idx_pneumonia = [x for x in range(n_pneumonia)]
    shuffle_idx_normal = [x for x in range(n_normal)]

    num_train_pneumonia = int((1 - validation_split) * n_pneumonia)
    num_val_pneumonia = n_pneumonia - num_train_pneumonia
    num_train_normal = int((1 - validation_split) * n_normal)
    num_val_normal = n_normal - num_train_normal


def build_model():
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(128, 128, 1)))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.RMSprop(lr=1e-4),
                  metrics=['acc', keras.metrics.Recall()])
    return model
