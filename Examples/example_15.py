# https://machinelearningmastery.com/how-to-develop-a-conditional-generative-adversarial-network-from-scratch/
import os
import keras
import tensorflow
import numpy
import matplotlib.pyplot as plt


(X_train, Y_train), (X_test, Y_test) = keras.datasets.fashion_mnist.load_data()
print('Train', X_train.shape, Y_train.shape)
print('Test', X_test.shape, Y_test.shape)
for i in range(100):
    plt.subplot(10, 10, i+1)
    plt.imshow(X_train[i], cmap='gray_r')
    plt.axis('off')
plt.show()


def define_discriminator(in_shape=(28, 28, 1)):
    model = tensorflow.keras.Sequential()
    # DownSample
    model.add(tensorflow.keras.layers.Conv2D(128, (3, 3), padding='same', strides=(2, 2), input_shape=in_shape))
    model.add(tensorflow.keras.layers.LeakyReLU(alpha=0.2))
    print('DownSample_1', model.output_shape)
    # DownSample
    model.add(tensorflow.keras.layers.Conv2D(128, (3, 3), padding='same', strides=(2, 2)))
    model.add(tensorflow.keras.layers.LeakyReLU(alpha=0.2))
    print('DownSample_2', model.output_shape)
    # Classifier
    model.add(tensorflow.keras.layers.Flatten())
    model.add(tensorflow.keras.layers.Dropout(0.4))
    model.add(tensorflow.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=tensorflow.keras.optimizers.Adam(lr=2e-4, beta_1=5e-1), loss='binary_crossentropy', metrics=['accuracy'])
    return model


# print(2e-4, 5e-1)

def define_generator(latent_dim):
    model = tensorflow.keras.Sequential()
    num_nodes = 7 * 7 * 128
    model.add(tensorflow.keras.layers.Dense(num_nodes, input_dim=latent_dim))
    print('latent_dim', model.output_shape)
    model.add(tensorflow.keras.layers.LeakyReLU(alpha=0.2))
    model.add(tensorflow.keras.layers.Reshape((7, 7, 128)))
    # UpSample
    model.add(tensorflow.keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(tensorflow.keras.layers.LeakyReLU(alpha=0.2))
    print('UpSample_1', model.output_shape)
    # UpSample
    model.add(tensorflow.keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(tensorflow.keras.layers.LeakyReLU(alpha=0.2))
    print('UpSample_2', model.output_shape)
    # Generate
    model.add(tensorflow.keras.layers.Conv2D(1, kernel_size=(7, 7), padding='same', activation='tanh'))
    print('Generate', model.output_shape)
    return model


# define_discriminator()
# define_generator(100)


def define_gan(generator, discriminator):
    discriminator.trainable = False
    model = tensorflow.keras.Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(optimizer=tensorflow.keras.optimizers.Adam(lr=2e-4, beta_1=5e-1), loss='binary_crossentropy')
    return model


def load_real_samples():
    X = numpy.expand_dims(X_train, axis=-1)
    X = X.astype('float32')
    X = (X - 127.5) / 127.5
    return X


def generate_real_samples(dataset, n_samples):
    xi = numpy.random.randint(0, dataset.shape[0], n_samples)
    X =dataset[xi]
    Y = numpy.ones((n_samples, 1))
    return X, Y


def generate_latent_points(latent_dim, n_samples):
    x_input = numpy.random.randn(n_samples * latent_dim)
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input


def generate_fake_samples(generator, latent_dim, n_samples):
    x_input = generate_latent_points(latent_dim, n_samples)
    X = generator.predict(x_input)
    Y = numpy.zeros((n_samples, 1))
    return X, Y


def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=2, n_batch=128):
    bat_per_epo = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    for i in range(n_epochs):
        for j in range(bat_per_epo):
            if j == 15:
                break
            X_real, Y_real = generate_real_samples(dataset, half_batch)
            d_loss1, _ = d_model.train_on_batch(X_real, Y_real)
            print('d_loss1', d_loss1)
            X_fake, Y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            d_loss2, _ = d_model.train_on_batch(X_fake, Y_fake)
            print('d_loss2', d_loss2)
            X_gan = generate_latent_points(latent_dim, n_batch)
            Y_gan = numpy.ones((n_batch, 1))
            gan_loss = gan_model.train_on_batch(X_gan, Y_gan)
            print('>%d, %d/%d, d1=%.3f, d2=%.3f, gan=%.3f' % (i+1, j+1, bat_per_epo, d_loss1, d_loss2, gan_loss))
    g_model.save('g_model.h5')


def show_plot(examples, n):
    for i in range(n * n):
        plt.subplot(n, n, i + 1)
        plt.axis('off')
        plt.imshow(examples[i, :, :, 0], cmap='gray_r')
    plt.show()


def define_c_discriminator(in_shape=(28, 28, 1), n_classes=10):
    in_label = tensorflow.keras.layers.Input(shape=(1, ))
    li = tensorflow.keras.layers.Embedding(n_classes, 50)(in_label)
    n_nodes = in_shape[0] * in_shape[1]
    li = tensorflow.keras.layers.Dense(n_nodes)(li)
    li = tensorflow.keras.layers.Reshape((in_shape[0], in_shape[1], 1))(li)
    print('li', li.shape)
    # li = tensorflow.expand_dims(li, axis=3)
    # print('li', li.shape)
    in_image = tensorflow.keras.layers.Input(shape=in_shape)
    print('in_image', in_image.shape)
    merge = tensorflow.keras.layers.Concatenate()([in_image, li])
    fe = tensorflow.keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same')(merge)
    fe = tensorflow.keras.layers.LeakyReLU(alpha=0.2)(fe)
    fe = tensorflow.keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same')(fe)
    fe = tensorflow.keras.layers.LeakyReLU(alpha=0.2)(fe)
    fe = tensorflow.keras.layers.Flatten()(fe)
    fe = tensorflow.keras.layers.Dropout(0.4)(fe)
    out_layer = tensorflow.keras.layers.Dense(1, activation='sigmoid')(fe)
    model = tensorflow.keras.models.Model([in_image, in_label], out_layer)
    model.compile(optimizer=tensorflow.keras.optimizers.Adam(lr=2e-4, beta_1=5e-1), loss='binary_crossentropy', metrics=['accuracy'])
    tensorflow.keras.utils.plot_model(model, to_file='model.png', show_shapes=True)
    return model


def define_c_generator(latent_dim, n_classes=10):
    in_label = tensorflow.keras.layers.Input(shape=(1, ))
    li = tensorflow.keras.layers.Embedding(n_classes, 50)(in_label)
    n_nodes = 7 * 7
    li = tensorflow.keras.layers.Dense(n_nodes)(li)
    li = tensorflow.keras.layers.Reshape((7, 7, 1))(li)
    in_lat = tensorflow.keras.layers.Input(shape=(latent_dim, ))
    n_nodes = 128 * 7 * 7
    gen = tensorflow.keras.layers.Dense(n_nodes)(in_lat)
    gen = tensorflow.keras.layers.LeakyReLU(alpha=0.2)(gen)
    gen = tensorflow.keras.layers.Reshape((7, 7, 128))(gen)
    merge = tensorflow.keras.layers.Concatenate()([gen, li])
    print('merge', merge.shape)
    gen = tensorflow.keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(merge)
    gen = tensorflow.keras.layers.LeakyReLU(alpha=0.2)(gen)
    gen = tensorflow.keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(gen)
    gen = tensorflow.keras.layers.LeakyReLU(alpha=0.2)(gen)
    out_layer = tensorflow.keras.layers.Conv2D(1, (7, 7), activation='tanh', padding='same')(gen)
    model = tensorflow.keras.models.Model([in_lat, in_label], out_layer)
    tensorflow.keras.utils.plot_model(model, to_file='model.png', show_shapes=True)
    return model


def define_c_gan(c_g_model, c_d_model):
    c_d_model.trainable = False
    gen_noise, gen_label = c_g_model.input
    print('gen_noise', gen_noise.shape, 'gen_label', gen_label.shape)
    gen_output = c_g_model.output
    print('gen_output', gen_output.shape)
    gan_output = c_d_model([gen_output, gen_label])
    print('gan_output', gan_output)
    model = tensorflow.keras.models.Model([gen_noise, gen_label], gan_output)
    model.compile(optimizer=tensorflow.keras.optimizers.Adam(lr=2e-4, beta_1=5e-1), loss='binary_crossentropy')
    tensorflow.keras.utils.plot_model(model, to_file='model.png', show_shapes=True)
    return model


def load_c_real_samples():
    X = tensorflow.expand_dims(X_train, axis=-1)
    X = X.astype('float32')
    X = (X - 127.5) / 127.5
    return [X, Y_train]


def generate_c_real_samples(dataset, n_samples):
    images, labels = dataset
    ix = numpy.random.randint(0, images.shape[0], n_samples)
    X, labels = images[ix], labels[ix]
    Y = numpy.ones((n_samples, 1))
    return X, Y


def generate_c_latent_points(latent_dim, n_samples, n_classes):
    X = numpy.random.randn(n_samples * latent_dim)
    Z = X.reshape(n_samples, latent_dim)
    labels = numpy.random.randint(0, n_classes, n_samples)
    return [Z, labels]


def generate_fake_samples(generator, latent_dim, n_samples):
    Z, labels = generate_latent_points(latent_dim, n_samples)
    images = generator.predict([Z, labels])
    Y = numpy.zeros((n_samples, 1))
    return [images, labels], Y



'''    
latent_dim = 100
discriminator = define_discriminator()
generator = define_generator(latent_dim)
gan_model = define_gan(generator, discriminator)
dataset = load_real_samples()
train(generator, discriminator, gan_model, dataset, latent_dim)
'''


'''
model = tensorflow.keras.models.load_model('g_model.h5')
latent_points = generate_latent_points(100, 100)
X = model.predict(latent_points)
show_plot(X, 10)
'''
# define_c_discriminator()
# define_c_generator(100)
define_c_gan(define_c_generator(100), define_c_discriminator())