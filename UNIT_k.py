# https://github.com/simontomaskarlsson/GAN-MRI/tree/master/UNIT
import os
import sys
import glob
import time
import datetime
import numpy
import keras
import keras.backend as k


class UNIT():
    def __init__(self, lr=1e-04, date_time_string_addition=''):
        self.channels = 1
        weight_decay = 1e-04 / 2
        nr_A_train_imgs = 1
        nr_B_train_imgs = 1
        nr_A_test_imgs = 1
        nr_B_test_imgs = None
        image_folder = 'UNIT_k/'

        self.A_train = None
        self.B_train = None
        self.img_height = 256
        self.img_width = 256
        self.img_shape = (self.img_height, self.img_width, self.channels)
        self.learning_rate = lr
        self.beta_1 = 0.5
        self.beta_2 = 0.999
        self.lambda_0 = 10
        self.lambda_1 = 1e-01
        self.lambda_2 = 100
        self.lambda_3 = self.lambda_1
        self.lambda_4 = self.lambda_2

        opt = keras.optimizers.Adam(lr=self.learning_rate, beta_1=self.beta_1, beta_2=self.beta_2)
        opt_stand_adam = keras.optimizers.Adam()

        self.super_simple = self.model_simple()
        self.super_simple.compile(optimizer=opt_stand_adam, loss='mae')

        self.discriminator_A = self.multi_model_discriminator('discriminator_A')
        self.discriminator_B = self.multi_model_discriminator('discriminator_B')
        for layer in self.discriminator_A.layers:
            if hasattr(layer, 'kernel_regularizer'):
                layer.kernel_regularizer = keras.regularizers.l2(weight_decay)
                layer.bias_regularizer = keras.regularizers.l2(weight_decay)
            if hasattr(layer, 'kernel_initializer'):
                layer.kernel_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=2e-02)
                layer.bias_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=2e-02)
        for layer in self.discriminator_B.layers:
            if hasattr(layer, 'kernel_regularizer'):
                layer.kernel_regularizer = keras.regularizers.l2(weight_decay)
                layer.bias_regularizer = keras.regularizers.l2(weight_decay)
            if hasattr(layer, 'kernel_initializer'):
                layer.kernel_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=2e-02)
                layer.bias_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=2e-02)
        self.discriminator_A.compile(optimizer=opt,
                                     loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'],
                                     loss_weights=[self.lambda_0, self.lambda_0, self.lambda_0])
        self.discriminator_B.compile(optimizer=opt,
                                     loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'],
                                     loss_weights=[self.lambda_0, self.lambda_0, self.lambda_0])

        self.encoder_A = self.model_encoder('encoder_A')
        self.encoder_B = self.model_encoder('encoder_B')
        self.shared_encoder = self.model_shared_encoder('shared_encoder')
        self.shared_decoder = self.model_shared_decoder('shared_decoder')

        self.generator_A = self.model_generator('generator_A')
        self.generator_B = self.model_generator('generator_B')

        img_A = keras.layers.Input(shape=self.img_shape)
        img_B = keras.layers.Input(shape=self.img_shape)
        encoded_img_A = self.encoder_A(img_A)
        encoded_img_B = self.encoder_B(img_B)
        shared_A = self.shared_encoder(encoded_img_A)
        shared_B = self.shared_encoder(encoded_img_B)
        out_shared_A = self.shared_decoder(shared_A)
        out_shared_B = self.shared_decoder(shared_B)

        out_Aa = self.generator_A(out_shared_A)
        out_Ba = self.generator_A(out_shared_B)
        out_Ab = self.generator_B(out_shared_A)
        out_Bb = self.generator_B(out_shared_B)
        guess_out_Ba = self.discriminator_A(out_Ba)
        guess_out_Ab = self.discriminator_B(out_Ab)

        cycle_encoded_img_A = self.encoder_A(out_Ba)
        cycle_encoded_img_B = self.encoder_B(out_Ab)
        cycle_shared_A = self.shared_decoder(cycle_encoded_img_A)
        cycle_shared_B = self.shared_decoder(cycle_encoded_img_B)
        cycle_out_shared_A = self.shared_decoder(cycle_shared_A)
        cycle_out_shared_B = self.shared_decoder(cycle_shared_B)
        cycle_Ab_Ba = self.generator_A(cycle_out_shared_B)
        cycle_Ba_Ab = self.generator_B(cycle_out_shared_A)

        self.discriminator_A.trainable = False
        self.discriminator_B.trainable = False
        self.encoder_generator = keras.Model(inputs=[img_A, img_B],
                                             outputs=[shared_A, shared_B, cycle_shared_A, cycle_shared_B, out_Aa, out_Bb, cycle_Ab_Ba, cycle_Ba_Ab, guess_out_Ba[0], guess_out_Ab[0], guess_out_Ba[1], guess_out_Ab[1], guess_out_Ba[2], guess_out_Ab[2]])
        for layer in self.encoder_generator.layers:
            if hasattr(layer, 'kernel_regularizer'):
                layer.kernel_regularizer = keras.regularizers.l2(weight_decay)
                layer.bias_regularizer = keras.regularizers.l2(weight_decay)
            if hasattr(layer, 'kernel_initializer'):
                layer.kernel_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=2e-02)
                layer.kernel_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=2e-02)

        self.encoder_generator.compile(optimizer=opt,
                                       loss=[self.vae_loss_cogan, self.vae_loss_cogan,self.vae_loss_cogan, self.vae_loss_cogan, 'mae', 'mae', 'mae', 'mae', 'binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'],
                                       loss_weights=[self.lambda_1, self.lambda_1, self.lambda_3, self.lambda_3, self.lambda_2, self.lambda_2, self.lambda_4, self.lambda_4, self.lambda_0, self.lambda_0, self.lambda_0, self.lambda_0, self.lambda_0, self.lambda_0])


    def resblk(self, x0, k):
        x = keras.layers.Conv2D(filter=k, kernel_size=3, strides=1, padding='same')(x0)
        x = keras.layers.BatchNormalization(axis=3, momentum=0.9, epsilon=1e-05, center=True)(x, training=True)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Conv2D(filters=k, kernel_size=3, strides=1, padding='same')(x)
        x = keras.layers.BatchNormalization(axis=3, momentum=0.9, epsilon=1e-05, center=True)(x, training=True)
        x = keras.layers.Dropout(0.5)(x, training=True)
        x = keras.layers.add([x, x0])
        return x

    def vae_loss_cogan(self, y_true, y_pred):
        y_pred_2 = k.square(y_pred)
        encoding_loss = k.mean(y_pred_2)
        return encoding_loss

    def model_discriminator(self, x):
        x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='same')(x)
        x = keras.layers.LeakyReLU(alpha=1e-02)(x)
        x = keras.layers.Conv2D(filters=128, kernel_size=3, strides=2, padding='same')(x)
        x = keras.layers.LeakyReLU(alpha=1e-02)(x)
        x = keras.layers.Conv2D(filters=256, kernel_size=3, strides=2, padding='same')(x)
        x = keras.layers.LeakyReLU(alpha=1e-02)(x)
        x = keras.layers.Conv2D(filters=512, kernel_size=3, strides=2, padding='same')(x)
        x = keras.layers.LeakyReLU(alpha=1e-02)(x)
        x = keras.layers.Conv2D(filters=1, kernel_size=1, strides=1)(x)
        prediction = keras.layers.Activation('sigmoid')(x)
        return prediction

    def multi_model_discriminator(self, name):
        x1 = keras.layers.Input(shape=self.img_shape)
        x2 = keras.layers.AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(x1)
        x4 = keras.layers.AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(x2)
        x1_out = self.model_discriminator(x1)
        x2_out = self.model_discriminator(x2)
        x4_out = self.model_discriminator(x4)
        return keras.Model(inputs=x1, outputs=[x1_out, x2_out, x4_out], name=name)

    def model_encoder(self, name):
        input_img = keras.layers.Input(shape=self.img_shape)
        x = keras.layers.ZeroPadding2D(padding=(3, 3))(input_img)
        x = keras.layers.Conv2D(filters=64, kernel_size=7, strides=1, padding='valid')(x)
        x = keras.layers.LeakyReLU(alpha=1e-02)(x)
        x = keras.layers.ZeroPadding2D(padding=(1, 1))(x)
        x = keras.layers.Conv2D(filters=128, kernel_size=3, strides=2, padding='valid')(x)
        x = keras.layers.LeakyReLU(alpha=1e-02)(x)
        x = keras.layers.ZeroPadding2D(padding=(1, 1))(x)
        x = keras.layers.Conv2D(filters=256, kernel_size=3, strides=2, padding='valid')(x)
        x = keras.layers.LeakyReLU(alpha=1e-02)(x)
        x = self.resblk(x, 256)
        x = self.resblk(x, 256)
        z = self.resblk(x, 256)
        return keras.Model(inputs=input_img, outputs=z, name=name)

    def model_shared_encoder(self, name):
        input_img = keras.layers.Input(shape=self.img_shape)
        x = self.resblk(input_img, 256)
        z = keras.layers.GaussianNoise(stddev=1)(x, training=True)
        return keras.Model(inputs=input_img, outputs=z, name=name)

    def model_shared_decoder(self, name):
        input_img = keras.layers.Input(shape=self.img_shape)
        x = self.resblk(input_img, 256)
        return keras.Model(inputs=input_img, outputs=x, name=name)

    def model_generator(self, name):
        input_img = keras.layers.Input(shape=self.img_shape)
        x = self.resblk(input_img, 256)
        x = self.resblk(x, 256)
        x = self.resblk(x, 256)
        x = keras.layers.Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding='same')(x)
        x = keras.layers.LeakyReLU(alpha=1e-02)(x)
        x = keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same')(x)
        x = keras.layers.LeakyReLU(alpha=1e-02)(x)
        x = keras.layers.Conv2DTranspose(filters=self.channels, kernel_size=1, strides=1, padding='valid')(x)
        z = keras.layers.Activation('tanh')(x)
        return keras.Model(inputs=input_img, outputs=z, name=name)

    def model_simple(self):
        input_img = keras.layers.Input(shape=self.img_shape)
        x = keras.layers.Conv2D(filters=256, kernel_size=1, strides=1, padding='same')(input_img)
        x = keras.layers.Activation('relu')(x)
        prediction = keras.layers.Conv2D(1, kernel_size=5, strides=1, padding='same')(x)
        return keras.Model(input=input_img, output=prediction)
