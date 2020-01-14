# https://github.com/affinelayer/pix2pix-tensorflow/blob/master/pix2pix.py
from __future__ import division, print_function
import os
import time
import tensorflow
import numpy
import collections


EPS = 1e-12
CROP_SIZE = 256

SEPARABLE_CONV = True
NGF = 64
NDF = 64
GAN_WEIGHT = 0.5
L1_WEIGHT = 0.5


Model = collections.namedtuple('Model', 'outputs, predict_real, predict_fake, discrim_loss, discrim_grads_and_vars, '
                                        'gen_loss_GAN, gen_loss_L1, gen_grads_and_vars, train')


def receptive_field_size(output_size=1, kernel_size=4, stride=2):
    return (output_size - 1) * stride + kernel_size


def preprocess(image):
    with tensorflow.name_scope('preprocess'):
        return image * 2 - 1


def deprocess(image):
    with tensorflow.name_scope('deprocess'):
        return (image + 1) / 2


def preprocess_lab(lab):
    with tensorflow.name_scope('preprocess_lab'):
        L_chan, a_chan, b_chan = tensorflow.unstack(lab, axis=2)
        return [L_chan / 50 - 1, a_chan / 110, b_chan / 110]


def deprocess_lab(L_chan, a_chan, b_chan):
    with tensorflow.name_scope('deprocess_lab'):
        return tensorflow.stack([(L_chan + 1) / 2 * 100, a_chan * 110, b_chan * 110], axis=3)

def check_image(image):
    assertion = tensorflow.assert_equal(tensorflow.shape(image)[-1], 3, message='image must have 3 color channels')
    with tensorflow.control_dependencies([assertion]):
        image = tensorflow.identity(image)
    if image.get_shape().ndims not in (3, 4):
        raise ValueError('Image must be either 3 or 4 dimensions')
    shape = list(image.get_shape())
    shape[-1] = 3
    image.set_shape(shape)
    return image


def lab_to_rgb(lab):
    with tensorflow.name_scope('lab_to_rgb'):
        lab = check_image(lab)
        lab_pixels = tensorflow.reshape(lab, [-1, 3])
        with tensorflow.name_scope('cielab_to_xyz'):
            lab_to_fxfyfz = tensorflow.constant([
                [1/116.0, 1/116.0, 1/116.0],
                [1/500.0, 0.0, 0.0],
                [0.0, 0.0, -1/200.0],
            ])
            fxfyfz_pixels = tensorflow.matmul(lab_pixels + tensorflow.constant([16.0, 0.0, 0.0]), lab_to_fxfyfz)
            epsilon = 6 / 29
            linear_mask = tensorflow.cast(fxfyfz_pixels <= epsilon, dtype=tensorflow.float32)
            exponential_mask = tensorflow.cast(fxfyfz_pixels > epsilon, dtype=tensorflow.float32)
            xyz_pixels = (3 * epsilon ** 2 * (fxfyfz_pixels - 4 / 29)) * linear_mask + (fxfyfz_pixels ** 3) * exponential_mask
            xyz_pixels = tensorflow.multiply(xyz_pixels, [0.950456, 1.0, 1.088754])
        with tensorflow.name_scope('xyz_to_srgb'):
            xyz_to_rgb = tensorflow.constant([
                [3.2404542, -0.9692660, 0.0556434],
                [-1.5371385, 1.8760108, -0.2040259],
                [-0.4985314, 0.0415560, 1.0572252]
            ])
            rgb_pixels = tensorflow.matmul(xyz_pixels, xyz_to_rgb)
            rgb_pixels = tensorflow.clip_by_value(rgb_pixels, 0.0, 1.0)
            linear_mask = tensorflow.cast(rgb_pixels <= 0.0031308, dtype=tensorflow.float32)
            exponential_mask = tensorflow.cast(rgb_pixels > 0.0031308, dtype=tensorflow.float32)
            srgb_pixels = (rgb_pixels * 12.92 * linear_mask) + ((rgb_pixels ** (1/2.4) * 1.055) - 0.055) * exponential_mask
        return tensorflow.reshape(srgb_pixels, tensorflow.shape(lab))


def augment(image, brightness):
    a_chan, b_chan = tensorflow.unstack(image, axis=3)
    L_chan = tensorflow.squeeze(brightness, axis=3)
    lab = deprocess_lab(L_chan, a_chan, b_chan)
    rgb = lab_to_rgb(lab)
    return rgb


def discrim_conv(batch_input, out_channels, stride):
    padded_input = tensorflow.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT')
    return tensorflow.layers.conv2d(padded_input, out_channels, kernel_size=4, strides=(stride, stride), padding='valid', kernel_initializer=tensorflow.random_normal_initializer(0, 2e-2))


def gen_conv(batch_input, out_channels):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tensorflow.random_normal_initializer(0, 2e-2)
    if SEPARABLE_CONV:
        return tensorflow.layers.separable_conv2d(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding='same', depthwise_initializer=initializer)
    else:
        return tensorflow.layers.conv2d(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding='same', kernel_initializer=initializer)


def gen_deconv(batch_input, out_channels):
    initializer = tensorflow.random_normal_initializer(0, 2e-2)
    if SEPARABLE_CONV:
        _b, _h, _w, _c = batch_input.shape
        resized_image = tensorflow.image.resize_images(batch_input, [_h*2, _w*2], method=tensorflow.image.ResizeMethod.NEAREST_NEIGHBOR)
        return tensorflow.layers.separable_conv2d(resized_image, out_channels, kernel_size=4, strides=(1, 1), padding='same', depthwise_initializer=initializer)
    else:
        return tensorflow.layers.conv2d_transpose(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding='same', kernel_initializer=initializer)


def lrelu(x, a):
    with tensorflow.name_scope('lrelu'):
        x = tensorflow.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tensorflow.abs(x)


def batch_norm(inputs):
    return tensorflow.layers.batch_normalization(inputs, axis=3, epsilon=1e-5, momentum=0.1, training=True, gamma_initializer=tensorflow.random_normal_initializer(1.0, 2e-2))


def create_generator(generator_inputs, generator_outputs_channels):
    layers = []
    with tensorflow.variable_scope('encoder_1'):
        output = gen_conv(generator_inputs, NGF)
        layers.append(output)

    layer_specs = [
        NGF * 2,
        NGF * 4,
        NGF * 8,
        NGF * 8,
        NGF * 8,
        NGF * 8,
        NGF * 8
    ]

    for out_channels in layer_specs:
        with tensorflow.variable_scope('encoder_%d' % (len(layers) + 1)):
            rectified = lrelu(layers[-1], 0.2)
            convolved = gen_conv(rectified, out_channels)
            output = batch_norm(convolved)
            layers.append(output)

    layer_specs = [
        (NGF * 8, 0.5),
        (NGF * 8, 0.5),
        (NGF * 8, 0.5),
        (NGF * 8, 0.0),
        (NGF * 4, 0.0),
        (NGF * 2, 0.0),
        (NGF, 0.0)
    ]

    num_encoder_layers = len(layers)
    for decoder_layer, (out_channel, dropout) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        with tensorflow.variable_scope('decoder_%d' % (skip_layer + 1)):
            if decoder_layer == 0:
                input = layers[-1]
            else:
                input = tensorflow.concat([layers[-1], layers[skip_layer]], axis=3)
            rectified = tensorflow.nn.relu(input)
            output = gen_deconv(rectified, out_channels)
            output = batch_norm(output)
            if dropout > 0.0:
                output = tensorflow.nn.dropout(output, keep_prob=1 - dropout)
            layers.append(output)
    # [batch, 128, 128, NGF * 2] => [batch, 256, 256, generator_output_channels]
    with tensorflow.variable_scope('decoder_1'):
        input = tensorflow.concat([layers[-1], layers[0]], axis=3)
        rectified = tensorflow.nn.relu(input)
        output = gen_deconv(rectified, generator_outputs_channels)
        output = tensorflow.tanh(output)
        layers.append(output)
    return layers[-1]


def create_model(inputs, targets):
    def create_discriminator(disc_inputs, disc_targets):
        n_layers = 3
        layers = []

        # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
        input = tensorflow.concat([disc_inputs, disc_targets], axis=3)
        with tensorflow.variable_scope('layer_1'):
            convolved = discrim_conv(input, NDF, stride=2)
            rectified = lrelu(convolved, 0.2)
            layers.append(rectified)

        for i in range(n_layers):
            with tensorflow.variable_scope('layer_%d' % (n_layers + 1)):
                out_channels = NDF * min(2**(i + 1), 8)
                stride = 1 if i == n_layers - 1 else 2
                convolved = discrim_conv(layers[-1], out_channels, stride=stride)
                normalized = batch_norm(convolved)
                rectified = lrelu(normalized, 0.2)
                layers.append(rectified)

        with tensorflow.variable_scope('layer_%d' % (len(layers) + 1)):
            convolved = discrim_conv(rectified, out_channels=1, stride=1)
            output = tensorflow.sigmoid(convolved)
            layers.append(output)
        return layers[-1]

    with tensorflow.variable_scope('generator'):
        out_channels = int(targets.get_shape()[-1])
        outputs = create_generator(input, out_channels)

    # create two discriminators: 1- real, 2- fake
    with tensorflow.name_scope('real_discriminator'):
        with tensorflow.variable_scope('discriminator'):
            predict_real = create_discriminator(inputs, targets)

    with tensorflow.name_scope('fake_discriminator'):
        with tensorflow.variable_scope('discriminator', reuse=True):
            predict_fake = create_discriminator(inputs, outputs)

    with tensorflow.name_scope('discriminator_loss'):
        discrim_loss = tensorflow.reduce_mean(-(tensorflow.log(predict_real + EPS) + tensorflow.log(1 - predict_fake + EPS)))

    with tensorflow.name_scope('generator_loss'):
        gen_loss_GAN = tensorflow.reduce_mean(-tensorflow.log(predict_fake + EPS))
        gen_loss_L1 = tensorflow.reduce_mean(tensorflow.abs(targets - outputs))
        gen_loss = gen_loss_GAN * GAN_WEIGHT + gen_loss_L1 * L1_WEIGHT

    with tensorflow.name_scope('discriminator_train'):
        discrim_tvars = [var for var in tensorflow.trainable_variables() if var.name.startsWith('discriminator')]
        discrim_optim = tensorflow.train.AdamOptimizer(2e-4, 0.5)
        discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
        discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

    with tensorflow.name_scope('generator_train'):
        with tensorflow.control_dependencies([discrim_train]):
            gen_tvars = [var for var in tensorflow.trainable_variables() if var.name.startsWith('generator')]
            gen_optim = tensorflow.train.AdamOptimizer(2e-4, 0.5)
            gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
            gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

    ema = tensorflow.train.ExponentialMovingAverage(decay=0.99)
    update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1])
    global_step = tensorflow.train.get_or_create_global_step()
    incr_global_step = tensorflow.assign(global_step, global_step + 1)

    return Model(
        predict_real=predict_real,
        predict_fake=predict_fake,
        discrim_loss=ema.average(discrim_loss),
        discrim_grads_and_vars=discrim_grads_and_vars,
        gen_loss_GAN=gen_loss_GAN,
        gen_loss_L1=gen_loss_L1,
        gen_grads_and_vars=gen_grads_and_vars,
        outputs=outputs,
        train=tensorflow.group(update_losses, incr_global_step, gen_train)
    )