# https://github.com/eriklindernoren/Keras-GAN/blob/master/pix2pix/pix2pix.py
# https://arxiv.org/pdf/1611.07004.pdf

import os
import glob
import nibabel
import NNKeras.nifti_utils
import keras
import numpy
import scipy


class Pix2Pix():
    def __init__(self):
        self.img_rows = 256
        self.img_cols = 256
        self.img_channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.img_channels)

        self.patch = int((self.img_rows / (2 ** 4)))  # patch (265 / 16) = 16
        self.disc_patch = (self.patch, self.patch, 1)

        self.generator_filters = 64
        self.discriminator_filters = 64

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse', optimizer=keras.optimizers.Adam(2e-4, beta_1=0.5, beta_2=0.999),
                                   metrics=['accuracy'])

        self.generator = self.build_generator()

        img_A = keras.layers.Input(shape=self.img_shape)
        img_B = keras.layers.Input(shape=self.img_shape)

        fake_A = self.generator(img_B)

        self.discriminator.trainable = False
        valid = self.discriminator([fake_A, img_B])

        self.combined = keras.Model(inputs=[img_A, img_B], outputs=[valid, fake_A])
        self.combined.compile(loss=['mse', 'mae'], loss_weights=[1, 100],
                              optimizer=keras.optimizers.Adam(lr=2e-4, beta_1=0.5, beta_2=0.999))

    def build_generator(self):

        def conv_2d(layer_input, filters, bn=True):
            init = keras.initializers.RandomNormal(stddev=2e-2)
            conv = keras.layers.Conv2D(filters=filters, kernel_size=(4, 4), strides=(2, 2), padding='same',
                                       kernel_initializer=init)(layer_input)
            if bn:
                conv = keras.layers.BatchNormalization()(conv)
            conv = keras.layers.LeakyReLU(0.2)(conv)
            return conv

        def deconv_2d(layer_input, layer_skip, filtes, dropout_rate=0.0):
            init = keras.initializers.RandomNormal(stddev=2e-2)
            deconv = keras.layers.Conv2DTranspose(filters=filtes, kernel_size=(4, 4), strides=(2, 2), padding='same',
                                                  kernel_initializer=init)(layer_input)
            deconv = keras.layers.BatchNormalization()(deconv)
            if dropout_rate != 0.0:
                deconv = keras.layers.Dropout(rate=dropout_rate)(deconv)
            deconv = keras.layers.ReLU()(deconv)
            deconv = keras.layers.Concatenate()([deconv, layer_skip])
            return deconv

        generator_input = keras.layers.Input(shape=self.img_shape)

        # DownSample / Encoding
        enc_1 = conv_2d(generator_input, filters=self.generator_filters * 1, bn=False)
        enc_2 = conv_2d(enc_1, filters=self.generator_filters * 2)
        enc_3 = conv_2d(enc_2, filters=self.generator_filters * 4)
        enc_4 = conv_2d(enc_3, filters=self.generator_filters * 8)
        enc_5 = conv_2d(enc_4, filters=self.generator_filters * 8)
        enc_6 = conv_2d(enc_5, filters=self.generator_filters * 8)
        enc_7 = conv_2d(enc_6, filters=self.generator_filters * 8)
        enc_8 = conv_2d(enc_7, filters=self.generator_filters * 8)

        # UpSampling / Decoding
        dec_1 = deconv_2d(enc_8, enc_7, filtes=self.generator_filters * 8)
        dec_2 = deconv_2d(dec_1, enc_6, filtes=self.generator_filters * 8)
        dec_3 = deconv_2d(dec_2, enc_5, filtes=self.generator_filters * 8)
        dec_4 = deconv_2d(dec_3, enc_4, filtes=self.generator_filters * 8)
        dec_5 = deconv_2d(dec_4, enc_3, filtes=self.generator_filters * 4)
        dec_6 = deconv_2d(dec_5, enc_2, filtes=self.generator_filters * 2)
        dec_7 = deconv_2d(dec_6, enc_1, filtes=self.generator_filters * 1)

        # Convolution is applied to map the number of the output channels
        dec_8 = keras.layers.Conv2DTranspose(filters=self.img_channels, kernel_size=(4, 4), strides=(2, 2),
                                             padding='same')(dec_7)
        dec_8 = keras.layers.Activation('tanh')(dec_8)
        return keras.Model(generator_input, dec_8)

    def build_discriminator(self, patch=16):

        def disc_layer(layer_input, filters, bn=True):
            init = keras.initializers.RandomNormal(stddev=2e-2)
            conv = keras.layers.Conv2D(filters=filters, kernel_size=(4, 4), strides=(2, 2), padding='same',
                                       kernel_initializer=init)(layer_input)
            if bn:
                conv = keras.layers.BatchNormalization()(conv)
            conv = keras.layers.LeakyReLU(0.2)(conv)
            return conv

        disc_input_1 = keras.layers.Input(shape=self.img_shape)  # (256, 256, 1)
        disc_input_2 = keras.layers.Input(shape=self.img_shape)  # (256, 256, 1)
        # Concatenate the two input images by channels
        disc_input = keras.layers.Concatenate(axis=-1)([disc_input_1, disc_input_2])  # (256, 256, 2)

        # Discriminator architectures according to the patch size
        disc_1 = disc_layer(disc_input, filters=self.discriminator_filters * 1, bn=False)  # (128, 128, 64)
        disc_2 = disc_layer(disc_1, filters=self.discriminator_filters * 2)  # (64, 64, 128)
        disc_3 = disc_layer(disc_2, filters=self.discriminator_filters * 4)  # (32, 32, 256)
        disc_4 = disc_layer(disc_3, filters=self.discriminator_filters * 8)  # (16, 16, 512)
        '''
        if patch == 70:
            disc_2 = disc_layer(disc_2, filters=self.discriminator_filters * 4)
            disc_2 = disc_layer(disc_2, filters=self.discriminator_filters * 8)
        '''
        disc_output = keras.layers.Conv2D(filters=1, kernel_size=(4, 4), strides=1, padding='same')(disc_4)

        return keras.Model([disc_input_1, disc_input_2], disc_output)

    def load_batch(self, batch_size=1):
        data_path = 'D:\Halimeh\Datasets\MSD\Task05_Prostate\imagesTr'
        files = glob.glob(data_path + '\p*')
        print('file_names', files)
        for file_path in files:
            nifti = nibabel.load(file_path)
            slices = nifti.get_data()
            print('slices', slices.shape)

            resized = NNKeras.nifti_utils.resize(nifti, 256, 256)
            resized = NNKeras.nifti_utils.scale_to_grayscale(resized)
            resized = (resized / 127.5) - 1.0
            print('resized', resized.min(), resized.max())
            for i in range(resized.shape[2]):
                m_1 = resized[..., i, 0]
                m_1 = m_1[None, :, :, None]
                m_2 = resized[..., i, 1]
                m_2 = m_2[None, :, :, None]
                yield m_1, m_2

    def train(self, epochs=2, batch_size=1, sample_interval=10):
        valid_label = numpy.ones((batch_size,) + self.disc_patch)
        fake_label = numpy.zeros((batch_size,) + self.disc_patch)

        for epoch in range(epochs):
            for batch_i, (img_A, img_B) in enumerate(self.load_batch()):
                print(img_A.shape, img_B.shape)

                # Train Discriminator
                fake_A = self.generator.predict(img_B)
                disc_loss_real = self.discriminator.train_on_batch([img_A, img_B], valid_label)
                disc_loss_fake = self.discriminator.train_on_batch([fake_A, img_B], fake_label)
                disc_loss = 0.5 * numpy.add(disc_loss_real, disc_loss_fake)
                print('disc_loss', disc_loss)

                # Train Generator
                self.discriminator.trainable = False
                gen_loss = self.combined.train_on_batch([img_A, img_B], [valid_label, img_A])
                self.discriminator.trainable = True
                print('gen_loss', gen_loss)

                print('Epoch %d, Batch %d, disc_loss %f, gen_loss %f' % (epoch, batch_i, disc_loss[0], gen_loss[0]))


pix2pix = Pix2Pix()
pix2pix.train()
