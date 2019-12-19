# https://www.tensorflow.org/tutorials/generative/cvae
from __future__ import print_function
import os
import time
import numpy
import tensorflow
import os
import glob
import matplotlib.pyplot as plt



(train_images, _), (test_images, _) = tensorflow.keras.datasets.mnist.load_data()
print('train_images', train_images.shape)
print('test_images', test_images.shape)

train_images = train_images.reshape(train_images.shape[0], train_images.shape[1], train_images.shape[2], 1).astype('float32')
test_images = test_images.reshape(test_images.shape[0], test_images.shape[1], test_images.shape[2], 1).astype('float32')
train_images = train_images / 255.0
test_images = test_images / 255.0

train_images[train_images >= 0.5] = 1.0
train_images[train_images < 0.5] = 0.0
test_images[test_images >= 0.5] = 1.0
test_images[test_images < 0.5] = 0.0


TRAIN_BUF = 60000
BATCH_SIZE = 100
TEST_BUF = 1000


train_dataset = tensorflow.data.Dataset.from_tensor_slices(train_images).shuffle(TRAIN_BUF).batch(BATCH_SIZE)
test_dataset = tensorflow.data.Dataset.from_tensor_slices(test_images).shuffle(TEST_BUF).batch(BATCH_SIZE)


class CVAE(tensorflow.keras.Model):
    def __init__(self, latent_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.inference_net = tensorflow.keras.Sequential(
            [
                tensorflow.keras.layers.InputLayer(input_shape=(28, 28, 1)),
                tensorflow.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
                tensorflow.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
                tensorflow.keras.layers.Flatten(),
                tensorflow.keras.layers.Dense(latent_dim + latent_dim),
            ]
        )
        self.generative_net = tensorflow.keras.Sequential(
            [
                tensorflow.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tensorflow.keras.layers.Dense(units=7*7*32, activation=tensorflow.nn.relu),
                tensorflow.keras.layers.Reshape(target_shape=(7, 7, 32)),
                tensorflow.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=(2, 2), padding='same', activation='relu'),
                tensorflow.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=(2, 2), padding='same', activation='relu'),
                tensorflow.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=(1, 1), padding='same')
            ]
        )

    @tensorflow.function
    def sample(self, eps=None):
        if eps is None:
            return tensorflow.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        print('x', x.shape)
        y = self.inference_net(x)
        print('y', y.shape)
        mean, logvar = tensorflow.split(self.inference_net(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        print('mean', type(mean), mean.shape)
        print('logvar', type(logvar), logvar.shape)
        eps = tensorflow.random.normal(shape=(100, 100))
        return eps * tensorflow.exp(logvar * 0.5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.generative_net(z)
        if apply_sigmoid:
            probs = tensorflow.sigmoid(logits)
            return probs
        return logits


optimizer = tensorflow.keras.optimizers.Adam(1e-4)


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tensorflow.math.log(2.0 * numpy.pi)
    return tensorflow.reduce_sum(
        -0.5 * ((sample - mean) ** 2.0 * tensorflow.exp(-logvar) + logvar + log2pi), axis=raxis
    )


@tensorflow.function
def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)

    cross_net = tensorflow.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = -tensorflow.reduce_sum(cross_net, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0.0, 0.0)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return tensorflow.reduce_mean(logpx_z + logpz - logqz_x)


@tensorflow.function
def compute_apply_gradients(model, x, optimizer):
    with tensorflow.GradientTape() as tape:
        loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


epochs = 1
latent_dim = 100
num_examples_to_generate = 16
random_vector_for_generation = tensorflow.random.normal(shape=[num_examples_to_generate, latent_dim])
model = CVAE(latent_dim)


def generate_images(model, epoch, test_input):
    prediction = model.sample(test_input)
    fig = plt.figure(figsize=(8, 8))
    with tensorflow.Session() as sess:
        sess.run(tensorflow.global_variables_initializer())
        prediction = sess.run(prediction)
        for i in range(prediction.shape[0]):
            plt.subplot(4, 4, i+1)
            print('prediction', type(prediction), prediction.shape)
            plt.imshow(prediction[i, :, :, 0], cmap='gray')
            plt.axis('off')
    plt.show()


# generate_images(model, 0, random_vector_for_generation)
for epoch in range(1, epochs + 1):
    start_time = time.time()
    for train_x in train_dataset:
        print('train_x', type(train_x), train_x.shape)
        compute_apply_gradients(model, train_x, optimizer)
    end_time = time.time()
    print('time', end_time - start_time)
    if epoch % 1 == 0:
        loss = tensorflow.keras.metrics.Mean()
        for test_x in test_dataset:
            loss(compute_loss(model, test_x))
        elbo = -loss.result()
        print('Epoch {} Test ELBO {} time {}'.format(epoch, elbo, end_time - start_time))
        generate_images(model, epoch, random_vector_for_generation)