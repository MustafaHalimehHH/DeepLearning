# https://machinelearningmastery.com/how-to-implement-pix2pix-gan-models-from-scratch-with-keras/

import keras


def discriminator(image_shape):
    init = keras.initializers.RandomNormal(stddev=2e-2)
    in_src_img = keras.layers.Input(shape=image_shape)
    in_target_img = keras.layers.Input(shape=image_shape)
    merged = keras.layers.Concatenate()([in_src_img, in_target_img])
    d = keras.layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(merged)
    d = keras.layers.LeakyReLU(alpha=0.2)(d)
    d = keras.layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = keras.layers.BatchNormalization()(d)
    d = keras.layers.LeakyReLU(alpha=0.2)(d)
    d = keras.layers.Conv2D(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = keras.layers.BatchNormalization()(d)
    d = keras.layers.LeakyReLU(alpha=0.2)(d)
    d = keras.layers.Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = keras.layers.BatchNormalization()(d)
    d = keras.layers.LeakyReLU(alpha=0.2)(d)
    d = keras.layers.Conv2D(512, (4, 4), padding='same', kernel_initializer=init)(d)
    d = keras.layers.BatchNormalization()(d)
    d = keras.layers.LeakyReLU(alpha=0.2)(d)
    d = keras.layers.Conv2D(1, (4, 4), padding='same', kernel_initializer=init)(d)
    patch_out = keras.layers.Activation('sigmoid')(d)
    model = keras.models.Model([in_src_img, in_target_img], patch_out)
    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=2e-4, beta_1=0.5), loss_weights=[0.5])
    return model

model = discriminator(image_shape=(256, 256, 3))
model.summary()


