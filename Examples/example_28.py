# https://github.com/eriklindernoren/Keras-GAN/blob/master/cyclegan/cyclegan.py
from __future__ import print_function, division
import os
import scipy
import keras
import numpy
import matplotlib.pyplot as plt


class InstanceNormalization(keras.layers.Layer):
    def __init__(self,
               axis=None,
               epsilon=1e-3,
               center=True,
               scale=True,
               beta_initializer='zeros',
               gamma_initializer='zeros',
               beta_regularizer=None,
               gamma_regularizer=None,
               beta_constraint=None,
               gamma_constraint=None,
               **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = keras.initializers.get(beta_initializer)
        self.gamma_initializer = keras.initializers.get(gamma_initializer)
        self.beta_regularizer = keras.regularizers.get(beta_regularizer)
        self.gamma_regularizer = keras.regularizers.get(gamma_regularizer)
        self.beta_constraint = keras.constraints.get(beta_constraint)
        self.gamma_constraint = keras.constraints.get(gamma_constraint)

    def build(self, input_shape):
        ndim = len(input_shape)
        if self.axis == 0:
            raise ValueError('Axis cannot be zero')
        if (self.axis is not None) and (ndim == 2):
            raise ValueError('Cannot specify axis for rank 1 tensor')
        self.input_spec = keras.layers.InputSpec(ndim=ndim)
        if self.axis is None:
            shape = (1, )
        else:
            shape = (input_shape[self.axis], )
        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.built = True

    def call(self, inputs, training=None):
        input_shape = keras.backend.int_shape(inputs)
        reduction_axes = list(range(0, len(input_shape)))
        if self.axis is not None:
            del reduction_axes[self.axis]
        del reduction_axes[0]

        mean = keras.backend.mean(inputs, reduction_axes, keepdims=True)
        stddev = keras.backend.std(inputs, reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs - mean) / stddev

        broadcast_shape = [1] * len(input_shape)
        if self.axis is not None:
            broadcast_shape[self.axis] = input_shape[self.axis]
        if self.scale:
            broadcast_gamma = keras.backend.reshape(self.gamma, broadcast_shape)
            normed = normed * broadcast_gamma
        if self.center:
            broadcast_beta = keras.backend.reshape(self.beta, broadcast_shape)
            normed = normed + broadcast_beta
        return normed

    def get_config(self):
        config = {
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': keras.initializers.serialize(self.beta_initializer),
            'gamma_initializer': keras.initializers.serialize(self.gamma_initializer),
            'beta_regularizer': keras.regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': keras.regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': keras.constraints.serialize(self.beta_constraint),
            'gamma_constraint': keras.constraints.serialize(self.gamma_constraint)
        }
        base_config = super(InstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class CycleGAN():
    def __init__(self):
        self.img_rows = 256
        self.img_cols = 256
        self.img_channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.img_channels)

        dataset_path = 'maps_256.npz'
        dataset = numpy.load(dataset_path)
        self.X1, self.X2 = dataset['arr_0'], dataset['arr_1']
        self.X1 = (self.X1 - 127.5) / 127.5
        self.X2 = (self.X2 - 127.5) / 127.5

        patch = int(self.img_rows / 2 ** 4)
        self.disc_patch = (patch, patch, 1)
        self.generator_filters = 32
        self.discriminator_filters = 64
        self.lambda_cycle = 10.0
        self.lambda_id = 0.1 * self.lambda_cycle

        self.d_A = self.build_discriminator()
        self.d_B = self.build_discriminator()
        self.d_A.compile(loss='mse', optimizer=keras.optimizers.Adam(2e-4, 0.5), metrics=['accuracy'])
        self.d_B.compile(loss='mse', optimizer=keras.optimizers.Adam(2e-4, 0.5), metrics=['accuracy'])

        self.g_AB = self.build_generator()
        self.g_BA = self.build_generator()

        img_A = keras.layers.Input(shape=self.img_shape)
        img_B = keras.layers.Input(shape=self.img_shape)

        fake_B = self.g_AB(img_A)
        fake_A = self.g_BA(img_B)

        reconst_A = self.g_BA(fake_B)
        reconst_B = self.g_AB(fake_A)

        img_A_id = self.g_BA(img_A)
        img_B_id = self.g_AB(img_B)

        self.d_A.trainable = False
        self.d_B.trainable = False

        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)

        self.combined = keras.Model(inputs=[img_A, img_B], outputs=[valid_A, valid_B, reconst_A, reconst_B, img_A_id, img_B_id])
        self.combined.compile(loss=['mse', 'mse', 'mae', 'mae', 'mae', 'mae'],
                              loss_weights=[1, 1, self.lambda_cycle, self.lambda_cycle, self.lambda_id, self.lambda_id],
                              optimizer=keras.optimizers.Adam(2e-4, 0.5))


    def build_generator(self):

        def conv_2d(layer_input, filters):
            conv = keras.layers.Conv2D(filters=filters, kernel_size=4, strides=2, padding='same')(layer_input)
            conv = keras.layers.LeakyReLU(0.2)(conv)
            conv = InstanceNormalization()(conv)
            return conv

        def deconv_2d(layer_input, skip_input, filters, dropout_rate=0.0):
            deconv = keras.layers.UpSampling2D(size=2)(layer_input)
            deconv = keras.layers.Conv2D(filters=filters, kernel_size=4, strides=1, padding='same', activation='relu')(deconv)
            if dropout_rate:
                deconv = keras.layers.Dropout(rate=dropout_rate)(deconv)
            deconv = InstanceNormalization()(deconv)
            deconv = keras.layers.Concatenate()([deconv, skip_input])
            return deconv

        d0 = keras.layers.Input(shape=self.img_shape)  # (128, 128, 1)

        # DownSampling
        d1 = conv_2d(d0, self.generator_filters)  # (64, 64, 32)
        d2 = conv_2d(d1, self.generator_filters * 2)  # (32, 32, 64)
        d3 = conv_2d(d2, self.generator_filters * 4)  # (16, 16, 128)
        d4 = conv_2d(d3, self.generator_filters * 8)  # (8, 8, 256)

        # UpSampling
        u1 = deconv_2d(d4, d3, self.generator_filters * 4)  # (16, 16, 128)
        u2 = deconv_2d(u1, d2, self.generator_filters * 2)  # (32, 32, 64)
        u3 = deconv_2d(u2, d1, self.generator_filters)  # (64, 64, 32)

        u4 = keras.layers.UpSampling2D(size=2)(u3)  # (128, 128, 1)
        output_img = keras.layers.Conv2D(filters=self.img_channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)  # (128, 128, 1)
        return keras.Model(d0, output_img)

    def build_discriminator(self):

        def d_layer(layer_input, filters, normalization=True):
            d = keras.layers.Conv2D(filters=filters, kernel_size=4, strides=2, padding='same')(layer_input)
            d = keras.layers.LeakyReLU(alpha=0.2)(d)
            if normalization:
                d = InstanceNormalization()(d)
            return d

        img = keras.layers.Input(shape=self.img_shape)
        d1 = d_layer(img, self.discriminator_filters, normalization=False)
        d2 = d_layer(d1, self.discriminator_filters * 2)
        d3 = d_layer(d2, self.discriminator_filters * 4)
        d4 = d_layer(d3, self.discriminator_filters * 8)

        validity = keras.layers.Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)
        return keras.Model(img, validity)

    def load_batch(self, X1, X2, batch_size=1):
        while(True):
            i = numpy.random.randint(0, X1.shape[0], batch_size)
            yield X1[i, ...], X2[i, ...]

    def train(self, epochs, batch_size=1, sample_interval=10):
        valid = numpy.ones((batch_size, ) + self.disc_patch)
        fake = numpy.zeros((batch_size, ) + self.disc_patch)

        for epoch in range(epochs):
            for batch_i, (imgs_A, imgs_B) in enumerate(self.load_batch(self.X1, self.X2, batch_size)):
                fake_B = self.g_AB.predict(imgs_A)
                fake_A = self.g_BA.predict(imgs_B)

                dA_loss_real = self.d_A.train_on_batch(imgs_A, valid)
                dA_loss_fake = self.d_A.train_on_batch(fake_A, fake)
                dA_loss = 0.5 * numpy.add(dA_loss_real, dA_loss_fake)

                dB_loss_real = self.d_B.train_on_batch(imgs_B, valid)
                dB_loss_fake = self.d_B.train_on_batch(fake_B, fake)
                dB_loss = 0.5 * numpy.add(dB_loss_real, dB_loss_fake)

                d_loss = 0.5 * numpy.add(dA_loss, dB_loss)

                g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, valid, imgs_A, imgs_B, imgs_A, imgs_B])

                print('[Epoch %d/%d] [Batch %d/%d]' % (epoch, epochs, batch_i, self.X1.shape[0] // batch_size))
                if batch_i % sample_interval == 0:
                    self.sample_images(epoch, batch_i)

    def sample_images(self, epoch, batch):
        imgs_A = None
        imgs_B = None

        fake_B = self.g_AB.predict(imgs_A)
        fake_A = self.g_BA.predict(imgs_B)

        reconstr_A = self.g_BA.predict(fake_B)
        reconstr_B = self.g_AB.predict(fake_A)

        gen_images = numpy.concatenate([imgs_A, fake_B, reconstr_A, imgs_B, fake_A, reconstr_B])
        gen_images = 0.5 * gen_images + 0.5
        titles = ['Original', 'Translated', 'Reconstructed']
        fig, axs = plt.sublplots(2, 3)
        cnt = 0
        for i in range(2):
            for j in range(3):
                axs[i, j].imshow(gen_images[cnt])
                axs[i, j].set_title(titles[j])
                axs[i, j].axis('off')


gan = CycleGAN()
gan.train(epochs=2)
