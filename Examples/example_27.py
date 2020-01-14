# https://github.com/jacobgil/keras-dcgan
import keras
import numpy
import math
import PIL


def generator_model():
    model = keras.Sequential()
    model.add(keras.layers.Dense(input_dim=100, output_dim=1024))
    model.add(keras.layers.Activation('tanh'))
    model.add(keras.layers.Dense(128*7*7))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('tanh'))
    model.add(keras.layers.Reshape((7, 7, 128), input_shape=(128*7*7, )))
    model.add(keras.layers.UpSampling2D(size=(2, 2))) # (9, 9, 128)
    model.add(keras.layers.Conv2D(64, (5, 5), padding='same'))
    model.add(keras.layers.Activation('tanh'))
    model.add(keras.layers.UpSampling2D(size=(2, 2)))
    model.add(keras.layers.Conv2D(1, (5, 5), padding='same'))
    model.add(keras.layers.Activation('tanh'))
    return model


def discriminator_model():
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(64, (5, 5), padding='same', input_shape=(28, 28, 1)))
    model.add(keras.layers.Activation('tanh'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))  # 26, 26
    model.add(keras.layers.Conv2D(128, (5, 5)))
    model.add(keras.layers.Activation('tanh'))
    model.add(keras.layers.UpSampling2D(size=(2, 2)))  # 28, 28
    model.add(keras.layers.Flatten())  # 28 * 28
    model.add(keras.layers.Dense(1024))
    model.add(keras.layers.Activation('tanh'))
    model.add(keras.layers.Dense(1))
    model.add(keras.layers.Activation('sigmoid'))
    return model


def generator_containing_discriminator(g, d):
    model = keras.Sequential()
    model.add(g)
    d.trainable = False
    model.add(d)
    return model


def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num) / width))
    shape = generated_images.shape[1: 3]
    image = numpy.zeros((height * shape[0], width * shape[1]), dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index / width)
        j = index % width
        image[i * shape[0]: (i + 1) * shape[0], j * shape[1]: (j + 1) * shape[1]] = \
            image[:, :, 0]
    return image


def train(batch_size=32, epochs=8):
    (X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()
    X_train = (X_train.astype(numpy.float32) / 127.5) - 1.0  # (X - 127.5) / 127.5
    X_train = X_train[:, :, :, None]
    X_test = X_test[:, :, :, None]
    print('X_train', X_train.shape)
    print('Y_train', Y_train.shape)

    d = discriminator_model()
    d.summary()
    g = generator_model()
    g.summary()
    d_on_g = generator_containing_discriminator(g, d)
    d_on_g.summary()

    g.compile(loss='binary_crossentropy', optimizer='SGD')
    d_on_g.compile(loss='binary_crossentropy', optimizer=keras.optimizers.SGD(lr=5e-4, momentum=0.9, nesterov=True))
    d.trainable = True
    d.compile(loss='binary_crossentropy', optimizer=keras.optimizers.SGD(lr=5e-4, momentum=0.9, nesterov=True))

    for epoch in range(epochs):
        for index in range(int(X_train.shape[0] / batch_size)):
            noise = numpy.random.uniform(-1.0, 1.0, size=(batch_size, 100))
            image_batch = X_train[index * batch_size: (index + 1) * batch_size, :, :, :]
            generated_images = g.predict(noise, verbose=0)
            if index % 20 == 0:
                pass

            x =numpy.concatenate((image_batch, generated_images))
            print('x concatenate shape', x.shape)
            y = [1] * batch_size + [0] * batch_size
            d_loss = d.train_on_batch(x, y)
            print('batch %d loss: %f' % (index, d_loss))
            noise = numpy.random.uniform(-1, 1, size=(batch_size, 100))
            d.trainable = False
            g_loss = d_on_g.train_on_batch(noise, [1] * batch_size)
            d.trainable = True
            print('batch %d g_loss: %f' % (index, g_loss))









train(batch_size=32)


