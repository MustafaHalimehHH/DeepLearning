# https://github.com/eriklindernoren/Keras-GAN/blob/master/cogan/cogan.py

import os
import sys
import numpy
import keras
import scipy
import scipy.ndimage
import matplotlib.pyplot as plt


class COGAN():
    def __init__(self):
        self.img_rows = 32
        self.img_cols = 32
        self.img_chs = 3
        self.img_shape = (self.img_rows, self.img_cols, self.img_chs)
        self.latent_dim = 100

        optimizer = keras.optimizers.Adam(2e-4, 0.5)

        self.d1, self.d2 = self.build_discriminator()
        self.d1.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        self.d2.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        self.g1, self.g2 = self.build_generator()

        z = keras.layers.Input(shape=(self.latent_dim, ))
        img1 = self.g1(z)
        img2 = self.g2(z)

        self.d1.trainable = False
        self.d2.trainable = False

        valid_1 = self.d1(img1)
        valid_2 = self.d2(img2)

        self.combined = keras.Model(z, [valid_1, valid_2])
        self.combined.compile(loss=['binary_crossentropy', 'binary_crossentropy'], optimizer=optimizer)


    def build_generator(self):
        model = keras.Sequential()
        model.add(keras.layers.Dense(256, input_dim=self.latent_dim))
        model.add(keras.layers.LeakyReLU(alpha=0.2))
        model.add(keras.layers.BatchNormalization(momentum=0.8))
        model.add(keras.layers.Dense(256))
        model.add(keras.layers.LeakyReLU(alpha=0.2))
        model.add(keras.layers.BatchNormalization(momentum=0.8))

        noise = keras.layers.Input(shape=(self.latent_dim))
        feature_repr = model(noise)

        # Generator_1
        g1 = keras.layers.Dense(1024)(feature_repr)
        g1 = keras.layers.LeakyReLU(alpha=0.2)(g1)
        g1 = keras.layers.BatchNormalization(momentum=0.8)(g1)
        g1 = keras.layers.Dense(numpy.prod(self.img_shape), activation='tanh')(g1)
        img1 = keras.layers.Reshape(self.img_shape)(g1)

        # Generator_2
        g2 = keras.layers.Dense(1024)(feature_repr)
        g2 = keras.layers.LeakyReLU(alpha=0.2)(g2)
        g2 = keras.layers.BatchNormalization(momentum=0.8)(g2)
        g2 = keras.layers.Dense(numpy.prod(self.img_shape), activation='tanh')(g2)
        img2 = keras.layers.Reshape(self.img_shape)(g2)

        return keras.Mode(noise, img1), keras.Model(noise, img2)

    def build_discriminator(self):
        img1 = keras.layers.Input(shape=self.img_shape)
        img2 = keras.layers.Input(shape=self.img_shape)

        model = keras.Sequential()
        model.add(keras.layers.Flatten(input_shape=self.img_shape))
        model.add(keras.layers.Dense(512))
        model.add(keras.layers.LeakyReLU(alpha=0.2))
        model.add(keras.layers.Dense(256))
        model.add(keras.layers.LeakyReLU(alpha=0.2))

        img1_embedding = model(img1)
        img2_embedding = model(img2)

        validity_1 = keras.layers.Dense(1, activation='sigmoid')(img1_embedding)
        validity_2 = keras.layers.Dense(1, activation='sigmoid')(img2_embedding)

        return keras.Model(img1, validity_1), keras.Model(img2, validity_2)

    def train(self, epochs, batch_size=128, sample_interval=50):
        X_train = None
        X_train = (X_train.astype(numpy.float32) - 127.5) / 127.5
        X_train = numpy.expand_dims(X_train, axis=3)
        border = int(X_train.shape[0] / 2)
        X1 = X_train[:border]
        X2 = X_train[border:]
        x2 = scipy.ndimage.rotate(X2, 90, axes=(1, 2))

        valid = numpy.ones((batch_size, 1))
        fake = numpy.zeros((batch_size, 1))

        for epoch in range(epochs):
            indx = numpy.random.randint(0, X1.shape[0], batch_size)
            imgs1 = X1[indx]
            imgs2 = X2[indx]

            noise = numpy.random.normal(0, 1, (batch_size, 100))

            gen_imgs1 = self.g1.predict(noise)
            gen_imgs2 = self.g2.predict(noise)

            d1_loss_real = self.d1.train_on_batch(imgs1, valid)
            d2_loss_real = self.d2.train_on_batch(imgs2, valid)
            d1_loss_fake = self.d1.train_on_batch(gen_imgs1, fake)
            d2_loss_fake = self.d2.train_on_batch(gen_imgs2, fake)

            d1_loss = 0.5 * numpy.add(d1_loss_real, d1_loss_fake)
            d2_loss = 0.5 * numpy.add(d2_loss_real, d2_loss_fake)

            g_loss = self.combined.train_on_batch(noise, [valid, valid])



