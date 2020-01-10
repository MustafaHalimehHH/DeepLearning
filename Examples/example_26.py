# https://www.kaggle.com/vikramtiwari/pix-2-pix-model-using-tensorflow-and-keras/notebook
import os
import datetime
import skimage
import scipy
import scipy.misc
import skimage.transform
import skimage.io
import numpy
import keras
import tensorflow
import matplotlib.pyplot as plt


class DataLoader():
    def __init__(self, dataset_name, img_res=(128, 128)):
        self.dataset_name = dataset_name
        self.img_res = img_res

    def read_image(self, path):
        return skimage.io.imread(path)

    def load_data(self, batch_size=1, is_testing=False):
        data_type = 'train' if not is_testing else 'test'
        path = None
        batch_images = numpy.random.choice(path, size=batch_size)

        imgs_A = []
        imgs_B = []
        for img_path in batch_images:
            img = self.imread(img_path)
            h, w, _ = img.shape
            _w = int(w / 2)
            img_A, img_B = img[:, :_w, :], img[:, _w:, :]
            img_A = skimage.transform.resize(img_A, self.img_res)
            img_B = skimage.transform.resize(img_B, self.img_res)

            if not is_testing and numpy.random.random() < 0.5:
                img_A = numpy.fliplr(img_A)
                img_B = numpy.fliplr(img_B)
            imgs_A.append(img_A)
            imgs_B.append(img_B)
        imgs_A = numpy.array(imgs_A) / 127.5 - 1.
        imgs_B = numpy.array(imgs_B) / 127.5 - 1.0
        return imgs_A, imgs_B

    def load_batch(self, batch_size=1, is_testing=False):
        data_typ = 'train' if not is_testing else 'test'
        path = None
        self.n_batches = int(len(path) / batch_size)

        for i in range(self.n_batches - 1):
            batch = path[i * batch_size: (i + 1) * batch_size]
            imgs_A, imgs_B = [], []
            for img in batch:
                img = self.read_image(img)
                h, w, _ = img.shape
                half_w = int(w / 2)
                img_A, img_B = img[:, :half_w, :], img[:, half_w:, :]
                img_A = skimage.transform.resize(img_A, self.img_res)
                img_B = skimage.transform.resize(img_B, self.img_res)
                if not is_testing and numpy.random.random() > 0.5:
                    img_A = numpy.fliplr(img_A)
                    img_B = numpy.fliplr(img_B)
                imgs_A.append(img_A)
                imgs_B.append(img_B)
            imgs_A = numpy.array(imgs_A) / 127.5 - 1.
            imgs_B = numpy.array(imgs_B) / 127.5 - 1.

            yield imgs_A, imgs_B


class Pix2Pis():
    def __init__(self):
        self.img_rows = 256
        self.img_cols = 256
        self.img_chs = 3
        self.img_shape = (self.img_rows, self.img_cols, self.img_chs)
        self.dataset_name = 'facades'
        self.data_loader = DataLoader(dataset_name=self.dataset_name, img_res=(self.img_rows, self.img_cols))

        patch = int(self.img_rows / 2 ** 4)
        self.disc_patch = (patch, patch, 1)
        self.gf = 64
        self.df = 64
        optimizer = keras.optimizers.Adam(lr=2e-4, beta_1=5e-1)
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        self.generator = self.build_generator()
        img_A = keras.layers.Input(shape=self.img_shape)
        img_B = keras.layers.Input(shape=self.img_shape)
        fake_A = self.generator(img_B)
        self.discriminator.trainable = False
        valid = self.discriminator([fake_A, img_B])
        self.combined = keras.models.Model(inputs=[img_A, img_B], outputs=[valid, fake_A])
        self.combined.compile(loss=['mse', 'mae'], loss_weights=[1, 100], optimizer=optimizer)

    def build_generator(self):
        def conv2d(input_layer, filters, f_size=4, bn=True):
            init = keras.initializers.RandomNormal(stddev=2e-2)
            c = keras.layers.Conv2D(filters=filters, kernel_size=f_size, strides=2, padding='same')(input_layer)
            c = keras.layers.LeakyReLU(alpha=0.2)(c)
            if bn:
                c = keras.layers.BatchNormalization()(c)
            return c

        def deconv2d(input_layer, skip_input, filters, f_size=4, dropout_rate=0):
            init = keras.initializers.RandomNormal(stddev=2e-2)
            d = keras.layers.UpSampling2D(size=2)(input_layer)
            d = keras.layers.Conv2D(filters=filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(
                d)
            if dropout_rate:
                d = keras.layers.Dropout(rate=dropout_rate)(d)
            d = keras.layers.BatchNormalization(momentum=0.8)(d)
            d = keras.layers.Concatenate()([d, skip_input])
            return d

        d0 = keras.layers.Input(shape=self.img_shape)  # (256, 256, 3)
        # DownSampling
        d1 = conv2d(d0, self.gf, bn=False)  # (128, 128, 64)
        d2 = conv2d(d1, self.gf * 2)  # (64, 64, 128)
        d3 = conv2d(d2, self.gf * 4)  # (32, 32, 256)
        d4 = conv2d(d3, self.gf * 8)  # (16, 16, 512)
        d5 = conv2d(d4, self.gf * 8)  # (8, 8, 512)
        d6 = conv2d(d5, self.gf * 8)  # (4, 4, 512)
        d7 = conv2d(d6, self.gf * 8)  # (2, 2, 512)
        # UpSampling
        u1 = deconv2d(d7, d6, self.gf * 8)  # (4, 4, 512)
        u2 = deconv2d(u1, d5, self.gf * 8)  # (8, 8, 512)
        u3 = deconv2d(u2, d4, self.gf * 8)  # (16, 16, 512)
        u4 = deconv2d(u3, d3, self.gf * 4)  # (32, 32, 256)
        u5 = deconv2d(u4, d2, self.gf * 2)  # (64, 64, 128)
        u6 = deconv2d(u5, d1, self.gf)  # (128, 128, 64)

        u7 = keras.layers.UpSampling2D(size=2)(u6)  # (256, 256, 64)
        output_img = keras.layers.Conv2D(self.img_chs, kernel_size=4, strides=1, padding='same', activation='tanh')(u7)

        model = keras.models.Model(d0, output_img)
        return model

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, bn=True):
            init = keras.initializers.RandomNormal(stddev=2e-2)
            d = keras.layers.Conv2D(filters, kernel_size=f_size, strides=2, padding='same', kernel_initializer=init)(layer_input)
            d = keras.layers.LeakyReLU(alpha=0.2)(d)
            if bn:
                d = keras.layers.BatchNormalization(momentum=0.8)(d)
            return d

        img_A = keras.layers.Input(shape=self.img_shape)  # (256, 256, 3)
        img_B = keras.layers.Input(shape=self.img_shape)  # (256, 256, 3)
        combined_imgs = keras.layers.Concatenate(axis=-1)([img_A, img_B])  # (256, 256, 6)
        d1 = d_layer(combined_imgs, self.df, bn=False)  # (128, 128, 64)
        d2 = d_layer(d1, self.df * 2)  # (64, 64, 128) # RF 46
        d3 = d_layer(d2, self.df * 4)  # (32, 32, 256) # RF 22
        d4 = d_layer(d3, self.df * 8)  # (16, 16, 512) # RF 10
        validity = keras.layers.Conv2D(1, kernel_size=4, strides=1, padding='same')(d4) # RF 4
        return keras.models.Model([img_A, img_B], validity)

    def train(self, epochs, batch_size=1, sample_interval=50):
        start_time = datetime.datetime.now()
        valid = numpy.ones((batch_size, ) + self.disc_patch)
        fake = numpy.zeros((batch_size, ) + self.disc_patch)

        for epoch in epochs:
            for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch(batch_size=batch_size)):
                fake_A = self.generator.predict(imgs_B)
                d_loss_real = self.discriminator.train_on_batch([imgs_A, imgs_B], valid)
                d_loss_fake = self.discriminator.train_on_batch([fake_A, imgs_B], fake)
                d_loss = 0.5 * numpy.add(d_loss_real, d_loss_fake)

                g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, imgs_A])
                elapsed_time = datetime.datetime.now() - start_time

                if batch_i % sample_interval == 0:
                    self.sample_images(epoch, batch_i)

    def sample_images(self, epoch, batch_i):
        r, c = 3, 3
        imgs_A, imgs_B = self.data_loader.load_batch(batch_size=3, is_testing=True)
        fake_A = self.generator.predict(imgs_B)
        gen_imgs = numpy.concatenate([imgs_B, fake_A, imgs_A])
        gen_imgs = 0.5 * gen_imgs + 0.5
        titles = ['Conditioned', 'Generated', 'Original']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt])
                axs[i, j].set_title(titles[i])
                axs[i, j].axis('off')


