# https://github.com/eriklindernoren/Keras-GAN/blob/master/bigan/bigan.py
from __future__ import print_function, division
import os
import numpy
import keras
import matplotlib.pyplot as plt


class BIGAN():
    def __init__(self, alpha=0.2, bn_momentum=0.8):
        self.alpha = alpha
        self.bn_momentum = bn_momentum
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        optimizer = keras.optimizers.Adam(lr=2e-4, beta_1=5e-1)

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=['binary_crossentropy'], optimizer=optimizer, metrics=['accuracy'])
        self.generator = self.build_generator()
        self.encoder = self.build_encoder()
        self.discriminator.trainable = False

        z = keras.layers.Input(shape=(self.latent_dim,))
        img_ = self.generator(z)

        img = keras.layers.Input(shape=self.img_shape)
        z_ = self.encoder(img)

        fake = self.discriminator([z, img_])
        valid = self.discriminator([z_, img])

        self.bigan_generator = keras.models.Model([z, img], [fake, valid])
        self.bigan_generator.compile(loss=['binary_crossentropy', 'binary_crossentropy'], optimizer=optimizer)

    def build_encoder(self):
        model = keras.Sequential()
        model.add(keras.layers.Flatten(input_shape=self.img_shape))
        model.add(keras.layers.Dense(512))
        model.add(keras.layers.LeakyReLU(alpha=self.alpha))
        model.add(keras.layers.BatchNormalization(momentum=self.bn_momentum))
        model.add(keras.layers.Dense(512))
        model.add(keras.layers.LeakyReLU(alpha=self.alpha))
        model.add(keras.layers.BatchNormalization(momentum=self.bn_momentum))
        model.add(keras.layers.Dense(self.latent_dim))

        model.summary()

        img = keras.layers.Input(shape=self.img_shape)
        z = model(img)
        return keras.models.Model(img, z)

    def build_generator(self):
        model = keras.Sequential()
        model.add(keras.layers.Dense(512, input_dim=self.latent_dim))
        model.add(keras.layers.LeakyReLU(alpha=self.alpha))
        model.add(keras.layers.BatchNormalization(momentum=self.bn_momentum))
        model.add(keras.layers.Dense(512))
        model.add(keras.layers.LeakyReLU(alpha=self.alpha))
        model.add(keras.layers.BatchNormalization(momentum=self.bn_momentum))
        model.add(keras.layers.Dense(numpy.prod(self.img_shape), activation='tanh'))
        model.add(keras.layers.Reshape(self.img_shape))

        model.summary()

        z = keras.layers.Input(shape=(self.latent_dim,))
        gen_img = model(z)
        return keras.models.Model(z, gen_img)

    def build_discriminator(self):
        z = keras.layers.Input(shape=(self.latent_dim,))
        img = keras.layers.Input(shape=self.img_shape)
        d_in = keras.layers.concatenate([z, keras.layers.Flatten()(img)])

        model = keras.layers.Dense(1024)(d_in)
        model = keras.layers.LeakyReLU(alpha=self.alpha)(model)
        model = keras.layers.Dropout(0.5)(model)
        model = keras.layers.Dense(1024)(model)
        model = keras.layers.LeakyReLU(alpha=self.alpha)(model)
        model = keras.layers.Dropout(0.5)(model)
        model = keras.layers.Dense(1024)(model)
        model = keras.layers.LeakyReLU(alpha=self.alpha)(model)
        model = keras.layers.Dropout(0.5)(model)
        validity = keras.layers.Dense(1, activation='sigmoid')(model)

        model.summary()
        return keras.models.Model([z, img], validity)

    def sample_interval(self, epoch):
        r, c = 5, 5
        z = numpy.random.normal(size=(25, self.latent_dim))
        gen_imgs = self.generator.predict(z)
        gen_imgs = 0.5 * (gen_imgs + 0.5)

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefie("images/mnist%d.png" % epoch)
        plt.close()

    def train(self, epochs, batch_size=128, sample_interval=50):
        (X_train, _), (_, _) = keras.datasets.mnist.load_data()
        X_train = (X_train.astype(numpy.float32) - 127.5) / 127.5
        X_train = numpy.expand_dims(X_train, axis=3)

        valid = numpy.ones((batch_size, 1))
        fake = numpy.zeros((batch_size, 1))

        for epoch in epochs:
            z = numpy.random.normal(size=(batch_size, self.latent_dim))
            imgs_ = self.generator.predict(z)

            idx = numpy.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]
            z_ = self.encoder.predict(imgs)

            d_loss_real = self.discriminator.train_on_batch([z_, imgs], valid)
            d_loss_fake = self.discriminator.train_on_batch([z, imgs_], fake)
            d_loss = 0.5 * numpy.add(d_loss_real, d_loss_fake)

            g_loss = self.bigan_generator.train_on_batch([z, imgs], [valid, fake])

            print("%d [D loss: %f, acc: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss[0]))

            if epoch % sample_interval == 0:
                self.sample_interval(epoch)
