# https://github.com/eriklindernoren/Keras-GAN/blob/master/ccgan/ccgan.py
from __future__ import print_function, division
import os
import keras
import numpy
import scipy
import scipy.misc
import skimage
import matplotlib.pyplot as plt


class CCGAN():
    def __init__(self):
        self.img_rows = 32
        self.img_cols = 32
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.mask_height = 10
        self.mask_width = 10
        self.num_classes = 10

        self.gf = 32
        self.df = 32

        optimizer = keras.optimizers.Adam(2e-4, 5e-1)
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=['mse', 'categorical_crossentropy'],
                                   loss_weights=[0.5, 0.5],
                                   metrics=['accuracy'],
                                   optimizer=optimizer)
        self.generator = self.build_generator()
        '''
        self.generator.compile(loss=[],
                               metrics=[],
                               optimizer=optimizer)
        '''
        masked_input = keras.layers.Input(shape=self.img_shape)
        gen_img = self.generator(masked_input)
        self.discriminator.trainable = False
        valid, _ = self.discriminator(gen_img)
        self.combined = keras.models.Model(masked_input, valid)
        self.combined.compile(loss=['mse'],
                              optimizer=optimizer)

    def build_generator(self):
        def conv2d(layer_input, filters, f_size=4, bn=True):
            d = keras.layers.Conv2D(filters=filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = keras.layers.LeakyReLU(alpha=0.2)(d)
            if bn:
                d = keras.layers.BatchNormalization(momentum=0.8)(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            u = keras.layers.UpSampling2D(size=2)(layer_input)
            u = keras.layers.Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = keras.layers.Dropout(dropout_rate)(u)
            u = keras.layers.BatchNormalization(momentum=8e-1)(u)
            u = keras.layers.Concatenate()([u, skip_input])
            return u

        img = keras.layers.Input(shape=self.img_shape)
        d1 = conv2d(img, self.gf, bn=False)
        d2 = conv2d(d1, self.gf * 2)
        d3 = conv2d(d2, self.gf * 4)
        d4 = conv2d(d3, self.gf * 8)

        u1 = deconv2d(d4, d3, self.gf * 4)
        u2 = deconv2d(u1, d2, self.gf * 2)
        u3 = deconv2d(u2, d1, self.gf)
        u4 = keras.layers.UpSampling2D(size=2)(u3)
        output_img = keras.layers.Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)
        return keras.models.Model(img, output_img)

    def build_discriminator(self):
        img = keras.layers.Input(shape=self.img_shape)
        model = keras.Sequential()
        model.add(keras.layers.Conv2D(64, kernel_size=4, strides=2, padding='same', input_shape=self.img_shape))
        model.add(keras.layers.LeakyReLU(alpha=0.8))
        model.add(keras.layers.Conv2D(128, kernel_size=4, strides=2, padding='same'))
        model.add(keras.layers.LeakyReLU(alpha=0.2))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Conv2D(256, kernel_size=4, strides=2, padding='same'))
        model.add(keras.layers.LeakyReLU(alpha=0.2))
        model.add(keras.layers.BatchNormalization(momentum=8e-1))
        model.summary()

        img = keras.layers.Input(shape=self.img_shape)
        features = model(img)
        validity = keras.layers.Conv2D(1, kernel_size=4, strides=1, padding='same')(features)
        label = keras.layers.Flatten()(features)
        label = keras.layers.Dense(self.num_classes + 1, activation='softmax')(label)
        return keras.models.Model(img, [validity, label])

    def mask_randomly(self, imgs):
        y1 = numpy.random.randint(0, self.img_rows - self.mask_height, imgs.shape[0])
        y2 = y1 + self.mask_height
        x1 = numpy.random.randint(0, self.img_rows - self.mask_height, imgs.shape[0])
        x2 = x1 + self.mask_width
        masked_imgs = numpy.empty_like(imgs)
        for i, img in enumerate(imgs):
            masked_img = img.copy()
            _y1, _y2, _x1, _x2 = y1[i], y2[i], x1[i], x2[i]
            masked_img[_y1: _y2, _x1: _x2, :] = 0
            masked_imgs[i] = masked_img
        return masked_imgs

    def train(self, epochs=6, batch_size=128, sample_interval=2):
        (X_train, Y_train), (_, _) = keras.datasets.mnist.load_data()
        # X_train = numpy.array([scipy.misc.imresize(x, [self.img_rows, self.img_cols]) for x in X_train])
        X_train = numpy.array([skimage.transform.resize(x, [self.img_rows, self.img_cols]) for x in X_train])
        X_train = (X_train.astype(numpy.float32) - 127.5) / 127.5
        X_train = numpy.expand_dims(X_train, axis=3)
        Y_train = Y_train.reshape(-1, 1)

        valid = numpy.ones((batch_size, 4, 4, 1))
        fake = numpy.zeros((batch_size, 4, 4, 1))

        for epoch in range(epochs):
            idx = numpy.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]
            labels = Y_train[idx]
            masked_imgs = self.mask_randomly(imgs)
            gen_imgs = self.generator.predict(masked_imgs)
            labels = keras.utils.to_categorical(labels, num_classes=self.num_classes + 1)
            fake_labels = keras.utils.to_categorical(numpy.full((batch_size, 1), self.num_classes), num_classes=self.num_classes + 1)
            d_loss_real = self.discriminator.train_on_batch(imgs, [valid, labels])
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, [fake, fake_labels])
            d_loss = 0.5 * numpy.add(d_loss_real, d_loss_fake)
            g_loss = self.combined.train_on_batch(masked_imgs, valid)
            print('%d [D loss: %f, op_acc: %.2f%%] [G loss: %f]' % (epoch, d_loss[0], 100 * d_loss[4], g_loss))

            if epoch % sample_interval == 0:
                idx = numpy.random.randint(0, X_train.shape[0], 6)
                imgs = X_train[idx]
                self.sample_images(epoch, imgs)
                self.save_model()

    def sample_images(self, epoch, imgs):
        r, c = 3, 6
        masked_imgs = self.mask_randomly(imgs)
        gen_imgs = self.generator.predict(masked_imgs)
        imgs = (imgs + 1.0) * 0.5
        masked_imgs = (masked_imgs + 1.0) * 0.5
        gen_imgs = (gen_imgs + 1.0) * 0.5
        gen_imgs = numpy.where(gen_imgs < 0, 0, gen_imgs)
        fig, axs = plt.subplots(r, c)
        for i in range(c):
            axs[0, i].imshow(imgs[i, :, :, 0], cmap='gray')
            axs[0, i].axis('off')
            axs[1, i].imshow(masked_imgs[i, :, :, 0], cmap='gray')
            axs[1, i].axis('off')
            axs[2, i].imshow(gen_imgs[i, :, :, 0], cmap='gray')
            axs[2, i].axis('off')
        fig.savefig('figimages%d.png' % epoch)
        plt.close()

    def save_model(self):
        def save(model, model_name):
            model_path = 'model_name[%s].json' % model_name
            weights_path = 'saved_model[%s]_weights.hdf5' % model_name
            options = {'file_arch': model_path,
                       'file_weight': weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])
        save(self.generator, 'ccgan_generator')
        save(self.discriminator, 'ccgan_discriminator')

print('Start')
ccgan = CCGAN()
ccgan.train(batch_size=32)
print('End')