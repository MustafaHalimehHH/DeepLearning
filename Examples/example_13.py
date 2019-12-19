# https://www.tensorflow.org/tutorials/generative/pix2pix
from __future__ import print_function, division, absolute_import, unicode_literals
import os
import numpy
import pandas
import keras
import tensorflow
import matplotlib.pyplot as plt
import pydot


_URL = 'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz'
path_to_zip = tensorflow.keras.utils.get_file('facades.tar.gz', origin=_URL, extract=True)
PATH = os.path.join(os.path.dirname(path_to_zip), 'facades/')

BUFFER_SIZE = 400
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256


def load(image_file):
    image = tensorflow.io.read_file(image_file)
    image = tensorflow.image.decode_jpeg(image, channels=3)
    print('image', image.shape)
    w = tensorflow.shape(image)[1]
    w = w // 2
    real_image = image[:, :w, :]
    input_image = image[:, w:, :]

    real_image = tensorflow.cast(real_image, tensorflow.float32)
    input_image = tensorflow.cast(input_image, tensorflow.float32)
    print('real_image', type(real_image), real_image.shape)
    print('input_image', type(input_image), input_image.shape)
    return input_image, real_image


def plot_input_real(input_image, real_image):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 8))
    with tensorflow.Session() as sess:
        sess.run(tensorflow.global_variables_initializer())
        i_img = sess.run(input_image)
        r_img = sess.run(real_image)
        ax1.imshow(i_img / 255.0)
        ax2.imshow(r_img / 255.0)
        plt.show()


def resize(input_image, real_image, height, width):
    input_image = tensorflow.image.resize(input_image, [height, width], method=tensorflow.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tensorflow.image.resize(real_image, [height, width], method=tensorflow.image.ResizeMethod.NEAREST_NEIGHBOR)
    print('resize:', input_image.shape, real_image.shape)
    return input_image, real_image


def random_crop(input_image, real_image):
    stacked_image = tensorflow.stack([input_image, real_image], axis=0)
    cropped_image = tensorflow.image.random_crop(stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])
    return cropped_image[0], cropped_image[1]


def normalize(input_image, real_image):
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1
    return input_image, real_image


@tensorflow.function()
def random_jitter(input_image, real_image):
    input_image, real_image = resize(input_image, real_image, 286, 286)
    input_image, real_image = random_crop(input_image, real_image)
    if numpy.random.uniform() > 0.5:
        input_image = tensorflow.image.flip_left_right(input_image)
        real_image = tensorflow.image.flip_left_right(real_image)
    return input_image, real_image


def load_image_train(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = random_jitter(input_image, real_image)
    input_image, real_image = normalize(input_image, real_image)
    return input_image, real_image


def load_image_test(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = resize(input_image, real_image, IMG_HEIGHT, IMG_WIDTH)
    input_image, real_image = normalize(input_image, real_image)
    return input_image, real_image





im, rm = load(PATH + 'train/100.jpg')
print('i', im.get_shape(), im.get_shape().ndims)
print('r', rm.get_shape(), rm.get_shape().ndims)
# plot_input_real(i, r)
plt.figure(figsize=(6, 6))
with tensorflow.Session() as sess:
    sess.run(tensorflow.global_variables_initializer())
    for i in range(4):
        i_j, r_j = random_jitter(im, rm)
        s_i_j = sess.run(i_j)
        s_r_j = sess.run(r_j)
        plt.subplot(4, 2, (2*i)+1)
        plt.imshow(s_i_j / 255.0)
        plt.axis('off')
        plt.subplot(4, 2, (2*i)+2)
        plt.imshow(s_r_j / 255.0)
        plt.axis('off')
    plt.show()


train_dataset = tensorflow.data.Dataset.list_files(PATH + 'train/*.jpg')
train_dataset = train_dataset.map(load_image_train, num_parallel_calls=tensorflow.data.experimental.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)

test_dataset = tensorflow.data.Dataset.list_files(PATH + 'test/*.jpg')
test_dataset = test_dataset.map(load_image_test)
test_dataset = test_dataset.batch(BATCH_SIZE)

OUTPUT_CHANNEL = 3


def downsample(filters, size, apply_batchnorm=True):
    initializer = tensorflow.random_normal_initializer(0.0, 0.02)
    result = tensorflow.keras.Sequential()
    result.add(tensorflow.keras.layers.Conv2D(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False))
    if apply_batchnorm:
        result.add(tensorflow.keras.layers.BatchNormalization())
    result.add(tensorflow.keras.layers.LeakyReLU())
    return result


def upsample(filters, size, apply_dropout=False):
    initializer = tensorflow.random_normal_initializer(0.0, 0.02)
    result = tensorflow.keras.Sequential()
    result.add(tensorflow.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False))
    result.add(tensorflow.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tensorflow.keras.layers.Dropout(0.5))
    result.add(tensorflow.keras.layers.ReLU())
    return result


def generator():
    inputs = tensorflow.keras.layers.Input(shape=[256, 256, 3])
    down_stack = [
        downsample(64, 4, apply_batchnorm=False),
        downsample(128, 4),
        downsample(256, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4)
    ]
    up_stack = [
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4),
        upsample(512, 4),
        upsample(256, 4),
        upsample(128, 4),
        upsample(64, 4)
    ]
    initializer = tensorflow.random_normal_initializer(0.0, 0.02)
    last = tensorflow.keras.layers.Conv2DTranspose(OUTPUT_CHANNEL, 4, strides=2, padding='same', kernel_initializer=initializer, activation='tanh')
    x = inputs
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tensorflow.keras.layers.Concatenate()([x, skip])
    x = last(x)
    return tensorflow.keras.Model(inputs=inputs, outputs=x)


LAMBDA = 100
loss_object = tensorflow.keras.losses.BinaryCrossentropy(from_logits=True)


def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tensorflow.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tensorflow.reduce_mean(tensorflow.abs(target - gen_output))
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)
    return total_gen_loss, gan_loss, l1_loss


def discriminator():
    initializer = tensorflow.random_normal_initializer(0.0, 0.02)
    inp = tensorflow.keras.layers.Input(shape=[256, 256, 3], name='input_image')
    tar = tensorflow.keras.layers.Input(shape=[256, 256, 3], name='target_image')
    x = tensorflow.keras.layers.concatenate([inp, tar])
    down1 = downsample(64, 4, False)(x)
    down2 = downsample(128, 4)(down1)
    down3 = downsample(256, 4)(down2)
    zero_pad1 = tensorflow.keras.layers.ZeroPadding2D()(down3)
    conv = tensorflow.keras.layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer, use_bias=False)(zero_pad1)
    batchnorm1 = tensorflow.keras.layers.BatchNormalization()(conv)
    leaky_relu = tensorflow.keras.layers.LeakyReLU()(batchnorm1)
    zero_pad2 = tensorflow.keras.layers.ZeroPadding2D()(leaky_relu)
    last = tensorflow.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(zero_pad2)
    return tensorflow.keras.Model(inputs=[inp, tar], outputs=last)


def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tensorflow.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(tensorflow.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss


def generate_images(model, test_input, tar):
    prediction = model(test_input, training=True)
    print('test_input', test_input[0].shape)
    print('tar', tar[0].shape)
    print('predication', prediction[0].shape)
    plt.figure(figsize=(8, 8))
    display_list = [test_input, tar, prediction]
    titles = ['Input Image', 'Ground Truth', 'Predicted Image']
    with tensorflow.Session() as sess:
        sess.run(tensorflow.global_variables_initializer())
        gg = sess.run(test_input[0])
        plt.imshow(gg[0] * 0.5 + 0.5)
        plt.axis('off')
        plt.show()
    '''
    for i in range(3):
        with tensorflow.Session() as sess:
            sess.run(tensorflow.global_variables_initializer())
            plt.subplot(1, 3, i+1)
            plt.title(titles[i])
            # gg = sess.run(display_list[i])
            # plt.imshow(gg[0] * 0.5 + 0.5)
            plt.axis('off')
    '''
    plt.show()


# inp, _ = resize(im, rm, IMG_HEIGHT, IMG_WIDTH)
inp, _ = random_crop(im, rm)
down_model = downsample(3, 4)
down_result = down_model(tensorflow.expand_dims(inp, 0))
print('down_result', down_result.shape, type(down_result))
up_model = upsample(3, 4)
up_result = up_model(down_result)
print('up_result', up_result.shape)
gen = generator()
# tensorflow.keras.utils.plot_model(gen, to_file='model.png', show_shapes=True)
gen_output = gen(inp[tensorflow.newaxis, ...], training=False)
with tensorflow.Session() as sess:
    sess.run(tensorflow.global_variables_initializer())
    g = sess.run(gen_output)
    plt.imshow(g[0, ...])
    plt.show()

disc = discriminator()
tensorflow.keras.utils.plot_model(disc, to_file='disc_model.png', show_shapes=True)
disc_out = disc([inp[tensorflow.newaxis, ...], gen_output], training=False)
with tensorflow.Session() as sess:
    sess.run(tensorflow.global_variables_initializer())
    g = sess.run(disc_out)
    plt.imshow(g[0, ..., -1], vmin=-20, vmax=20, cmap='RdBu_r')
    plt.colorbar()
    plt.show()

generator_optimizer = tensorflow.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tensorflow.keras.optimizers.Adam(2e-4, beta_1=0.5)
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tensorflow.train.Checkpoint(generator_optimizer=generator_optimizer,
                                         discriminator_optimizer=discriminator_optimizer,
                                         generator=gen,
                                         discriminator=disc)
'''
for example_input, example_target in test_dataset.take(1):
    print('example_input', type(example_input), type(example_target))
    generate_images(gen, example_input, example_target)
    break
'''
# example_input, example_target = tensorflow.data.experimental.get_single_element(test_dataset)
# generate_images(gen, example_input, example_target)

EPOCHS = 1
import datetime
log_dir = 'logs/'
summary_writer = tensorflow.summary.create_file_writer(
    log_dir + 'fit/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
)


@tensorflow.function
def train_step(input_image, target, epoch):
    with tensorflow.GradientTape() as gen_tape, tensorflow.GradientTape() as disc_tape:
        gen_output = gen(input_image, training=True)
        disc_real_output = disc([input_image, target], training=True)
        disc_generated_output = disc([input_image, gen_output], training=True)
        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
    generator_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, disc.trainable_variables)
    generator_optimizer.apply_gradients(zip(generator_gradients, gen.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, disc.trainable_variables))
    with summary_writer.as_default():
        tensorflow.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
        tensorflow.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
        tensorflow.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
        tensorflow.summary.scalar('disc_loss', disc_loss, step=epoch)


import time


def fit(train_ds, epochs, test_ds):
    for epoch in range(epochs):
        start = time.time()
        for example_input, example_target in test_ds.take(1):
            generate_images(gen, example_input, example_target)
        print("EPOCH", epoch)

        for n, (input_image, target) in train_ds.enumerate():
            print('.', end='')
            if (n+1) % 100 == 0:
                print()
            train_step(input_image, target, epoch)
        print()

        if (epoch + 1) % 20 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
        print('Time taken for epoch {} is {} sec\n'.format(epoch+1), time.time()-start)
        checkpoint.save(file_prefix=checkpoint_prefix)


