# https://www.tensorflow.org/tutorials/generative/dcgan
from __future__ import absolute_import, print_function, division, unicode_literals
import os
import tensorflow
import glob
import imageio
import numpy
import PIL
import time
import matplotlib.pyplot as plt


print('tensorflow', tensorflow.__version__)

(train_images, train_labels), (_, _) = tensorflow.keras.datasets.mnist.load_data()
print('train_images', train_images.shape)
print('train_labels', train_labels.shape)

# add channel dimension
# train_images = train_images.reshape(train_images.shape[0], train_images.shape[1], train_images.shape[2], 1).astype('float32')
train_images = numpy.expand_dims(train_images, axis=3).astype('float32')
print('train_images', train_images.shape)
train_images = (train_images - 127.5) / 127.5  # [-1, 1]

BUFFER_SIZE = 60000
BATCH_SIZE = 256

train_dataset = tensorflow.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


def make_generator_model():
    model = tensorflow.keras.Sequential()
    model.add(tensorflow.keras.layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)))
    model.add(tensorflow.keras.layers.BatchNormalization())
    model.add(tensorflow.keras.layers.LeakyReLU())

    model.add(tensorflow.keras.layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)  # None is the batch size

    model.add(tensorflow.keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(tensorflow.keras.layers.BatchNormalization())
    model.add(tensorflow.keras.layers.LeakyReLU())

    model.add(tensorflow.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(tensorflow.keras.layers.BatchNormalization())
    model.add(tensorflow.keras.layers.LeakyReLU())

    model.add(tensorflow.keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model


generator = make_generator_model()
noise = tensorflow.random.normal([1, 100])
print('noise', type(noise), noise.shape, noise.dtype)
generated_image = generator(noise, training=False)
print('generated_image', type(generated_image), generated_image.shape, generated_image.dtype)
'''
exit(1)
sess = tensorflow.Session()
c = generated_image
with sess.as_default():
    # arr = numpy.ndarray(generated_image.eval())
    print(type(tensorflow.constant([1, 2, 3])))
    print(type(tensorflow.constant([1, 2, 3]).eval()))
    print('START')
    arr = c.eval()
    print('END')
    print('arr', arr.shape)
    plt.imshow(arr[0, :, :, 0], cmap='gray')
    plt.show()
'''


def make_discriminator_model():
    model = tensorflow.keras.Sequential()
    model.add(tensorflow.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
    model.add(tensorflow.keras.layers.LeakyReLU())
    model.add(tensorflow.keras.layers.Dropout(0.3))
    assert model.output_shape == (None, 14, 14, 64)

    model.add(tensorflow.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(tensorflow.keras.layers.LeakyReLU())
    model.add(tensorflow.keras.layers.Dropout(0.3))
    assert model.output_shape == (None, 7, 7, 128)

    model.add(tensorflow.keras.layers.Flatten())
    model.add(tensorflow.keras.layers.Dense(1))

    return model


discriminator = make_discriminator_model()
decision = discriminator(generated_image)
# print('Discrimination')
# print(decision, type(decision), decision.shape)
# a = tensorflow.print(decision)
# print('a', a)

cross_entropy = tensorflow.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tensorflow.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tensorflow.ones_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tensorflow.ones_like(fake_output), fake_output)


generator_optimizer = tensorflow.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tensorflow.keras.optimizers.Adam(1e-4)

check_point_dir = './training_checkpoints'
check_point_prefix = os.path.join(check_point_dir, 'ckpt')
check_point = tensorflow.train.Checkpoint(
    generator_optimizer=generator_optimizer,
    discriminator_optimizer=discriminator_optimizer,
    generator=generator,
    discriminator=discriminator
)
EPOCHS = 2
noise_dim = 100
num_examples_to_generate = 16

seed = tensorflow.random.normal([num_examples_to_generate, noise_dim])


@tensorflow.function
def train_step(images):
    noise = tensorflow.random.normal([BATCH_SIZE, noise_dim])

    with tensorflow.GradientTape() as gen_tape, tensorflow.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def train(dataset, epochs):
    print('dataset', tensorflow.data.experimental.cardinality(dataset))
    for epoch in range(epochs):
        start = time.time()
        for image_batch in dataset:
            print('image_batch', image_batch.shape)
            # train_step(image_batch)


train(train_dataset, EPOCHS)