# https://github.com/tdeboissiere/DeepLearningImplementations/blob/master/pix2pix/src/model/models.py
import os
import numpy
import keras
import keras.backend as K


def minb_disc(x):
    diffs = K.expand_dims(x, 3) - K.expand_dims(K.premute_dimensions(x, [1, 2, 0]), 0)
    abs_diffs = K.sum(K.abs(diffs), 2)
    x = K.sum(K.exp(-abs_diffs), 2)
    return x


def lambda_output(input_shape):
    return input_shape[:2]


def conv_block_unet(x, f, name, bn_mode, bn_axis, bn=True, strides=(2, 2)):
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    x = keras.layers.Conv2D(f, (3, 3), strides=strides, name=name, padding='same')(x)
    if bn:
        x = keras.layers.BatchNormalization(axis=bn_axis)(x)
    return x


def up_conv_block_unet(x, x2, f, name, bn_mode, bn_axis, bn=True, dropout=False):
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.UpSampling2D(size=(2, 2))(x)
    x = keras.layers.Conv2D(f, (3, 3), name=name, padding='same')(x)
    if bn:
        x = keras.layers.BatchNormalization(axis=bn_axis)
    if dropout:
        x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Concatenate(axis=bn_axis)([x, x2])
    return x


def deconv_block_unet(x, x2, f, h, w, batch_size, name, bn_mode, bn_axis, bn=True, dropout=False):
    o_shape = (batch_size, h*2, w*2, f)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Deconv2D(f, (3, 3), output_shape=o_shape, strides=(2, 2), padding='same')(x)
    if bn:
        x = keras.layers.BatchNormalization(axis=bn_axis)(x)
    if dropout:
        x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Concatenate(axis=bn_axis)([x, x2])
    return x


def generator_unet_upsampling(img_dim, bn_mode, model_name='generator_unet_upsampling'):
    nb_filters = 64
    if K.image_dim_ordering() == 'channels_first':
        bn_axis = 1
        nb_channels = img_dim[0]
        min_s = min(img_dim[1:])
    else:
        bn_axis = -1
        bn_channels = img_dim[-1]
        min_s = min(img_dim[:-1])

    unet_input = keras.layers.Input(shape=img_dim, name='unet_input')
    nb_conv = int(numpy.floor(numpy.log(min_s) / numpy.log(2)))
    list_nb_filters = [nb_filters * min(8, (2 ** i)) for i in range(nb_conv)]
    list_encoder = [keras.layers.Conv2D(list_nb_filters[0], (3, 3), strides=(2, 2), padding='same', name='unet_conv2D_1')(unet_input)]
    for i, f in enumerate(list_nb_filters[1:]):
        name = 'unet_conv2D_%' % (i + 2)
        conv = conv_block_unet(list_encoder[-1], f, name, bn_mode, bn_axis)
        list_encoder.append(conv)

    list_nb_filters = list_nb_filters[:-2][::-1]
    if len(list_nb_filters) < nb_conv - 1:
        list_nb_filters.append(nb_filters)
    list_decoder = [up_conv_block_unet(list_encoder[-1], list_nb_filters[-2], list_nb_filters[0], 'unet_upconv2D_1', bn_mode, bn_axis, dropout=True)]
    for i, f in enumerate(list_nb_filters[1:]):
        name = 'unet_upconv2D_%s' % (i + 2)
        if i < 2:
            d = True
        else:
            d = False
        conv = up_conv_block_unet(list_decoder[-1], list_encoder[-(i + 3)], f, name, bn_mode, bn_axis, dropout=d)
        list_decoder.append(conv)

    x = keras.layers.Activation('relu')(list_decoder[-1])
    x = keras.layers.UpSampling2D(size=(2, 2))(x)
    x = keras.layers.Conv2D(nb_channels, (3, 3), name='last_conv', padding='same')(x)
    x = keras.layers.Activation('tanh')(x)

    generator_unet = keras.models.Model(inputs=[unet_input], outputs=[x])
    return generator_unet


def generator_unet_deconv(img_dim, bn_mode, batch_size, model_name='generator_unet_deconv'):
    assert K.backend() == 'tensorflow', 'Not implemented with others'

    nb_filters = 64
    bn_axis = -1
    h, w, nb_channels = img_dim
    min_s = min(img_dim[:-1])

    unet_input = keras.layers.Input(shape=img_dim, name='unet_input')

    nb_conv = int(numpy.floor(numpy.log(min_s) / numpy.log(2)))
    list_nb_filters = [nb_filters * min(8, (2 ** i)) for i in range(nb_conv)]
    list_encoder = [keras.layers.Conv2D(list_nb_filters[0], (3, 3), strides=(2, 2), name='unet_conv2D_1', padding='same')(unet_input)]
    h, w = h / 2, w / 2
    for i, f in enumerate(list_nb_filters[1:]):
        name = 'unet_conv2D_%s' % (i + 2)
        conv = conv_block_unet(list_encoder[-1], f, name, bn_mode, bn_axis)
        list_encoder.append(conv)
        h, w = h / 2, w / 2

    list_nb_filters = list_nb_filters[:-1][::-1]
    if len(list_nb_filters) < nb_conv - 1:
        list_nb_filters.append(nb_filters)

    list_decoder = [deconv_block_unet(list_encoder[-1], list_encoder[-2], list_nb_filters[0], h, w, batch_size, 'unet_upconv2D_1', bn_mode, bn_axis, dropout=True)]
    h, w = h * 2, w * 2
    for i, f in enumerate(list_nb_filters[1:]):
        name = 'unet_upconv2D_%s' % (i + 2)
        if i < 2:
            d = True
        else:
            d = False
        conv = deconv_block_unet(list_decoder[-1], list_encoder[-(i + 3)], f, h, w, batch_size, name, bn_mode, bn_axis, dropout=d)
        list_decoder.append(conv)
        h, w = h * 2, w * 2

    x = keras.layers.Activation('relu')(list_decoder[-1])
    o_shape = (batch_size, ) + img_dim
    x = keras.layers.Deconv2D(nb_channels, (3, 3), output_shape=o_shape, strides=(2, 2), padding='same')(x)
    x = keras.layers.Activation('tanh')(x)

    generator_unet = keras.model.Model(inputs=[unet_input], outputs=[x])
    return generator_unet


def DCGAN_discriminator(img_dim, nb_patch, bn_mode, model_name='DCGAN_discriminator', use_mbd=True):
    list_input = [keras.layers.Input(shape=img_dim, name='disc_input_%s' % i) for i in range(nb_patch)]
    if K.image_dim_ordering() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = -1

    nb_filters = 64
    nb_conv = int(numpy.floor(numpy.log(img_dim[1]) / numpy.log(2)))
    list_filters = [nb_filters * min(8, (2 ** i)) for i in range(nb_conv)]

    x_input = keras.layers.Input(shape=img_dim, name='discriminator_input')
    x = keras.layers.Conv2D(list_filters[0], (3, 3), strides=(2, 2), name='disc_conv2d_1', padding='same')(x_input)
    x = keras.layers.BatchNormalization(axis=bn_axis)(x)
    x = keras.layers.LeakyReLU(0.2)(x)

    for i, f in enumerate(list_filters[1:]):
        name = 'disc_conv2d%s' % (i + 2)
        x = keras.layers.Conv2D(f, (3, 3), strides=(2, 2), name=name, padding='same')(x)
        x = keras.layers.BatchNormalization(axis=bn_axis)(x)
        x = keras.layers.LeakyReLU(0.2)(x)

    x_flatt = keras.layers.Flatten()(x)
    x = keras.layers.Dense(2, activation='softmax', name='disc_dense')(x_flatt)

    PatchGAN = keras.models.Model(inputs=[x_input], outputs=[x, x_flatt], name='PatchGAN')
    print('PatchGAN summary')
    PatchGAN.summary()

    x = [PatchGAN(patch)[0] for patch in list_input]
    x_mbd = [PatchGAN(patch)[1] for patch in list_input]

    if len(x) > 1:
        x = keras.layers.Concatenate(axis=bn_axis)(x)
    else:
        x = x[0]

    if use_mbd:
        if len(x_mbd) > 1:
            x_mbd = keras.layers.Concatenate(axis=bn_axis)(x_mbd)
        else:
            x_mbd = x_mbd[0]
        num_kernels = 100
        dim_per_kernel = 5
        M = keras.layers.Dense(num_kernels * dim_per_kernel, use_bias=False, activation=None)
        MBD = keras.layers.Lambda(minb_disc, output_shape=lambda_output)
        x_mbd = M(x_mbd)
        x_mbd = keras.layers.Reshape((num_kernels, dim_per_kernel))(x_mbd)
        x_mbd = MBD(x_mbd)
        x = keras.layers.Concatenate(axis=bn_axis)([x, x_mbd])

    x_out = keras.layers.Dense(2, activation='softmax', name='disc_output')(x)
    discriminator_model = keras.models.Model(inputs=list_input, outputs=[x_out], name=model_name)
    return discriminator_model


def DCGAN(generator, discriminator_model, img_dim, patch_size, image_dim_ordering):
    gen_input = keras.layers.Input(shape=img_dim, name='DCGAN_input')
    generated_image = generator(gen_input)
    if image_dim_ordering == 'channels_first':
        h, w = img_dim[1:]
    else:
        h, w = img_dim[:-1]
    ph, pw = patch_size

    list_row_idx = [(i * ph, (i + 1) * ph) for i in range(h // ph)]
    list_col_idx = [(i * pw, (i + 1) * pw) for i in range(w // pw)]

    list_gen_patch = []
    for row_idx in list_row_idx:
        for col_idx in list_col_idx:
            if image_dim_ordering == 'channels_last':
                x_patch = keras.layers.Lambda(lambda z: z[:, row_idx[0]: row_idx[1], col_idx[0]: col_idx[1], :])(generated_image)
            else:
                x_patch = keras.layers.Lambda(lambda z: z[:, :, row_idx[0]: row_idx[1], col_idx[0]: col_idx[1]])(generated_image)
            list_gen_patch.append(x_patch)
    DCGAN_output = discriminator_model(list_gen_patch)
    DCGAN = keras.models.Model(inputs=[gen_input], outputs=[generated_image, DCGAN_output], name='DCGAN')
    return DCGAN


def load(model_name, img_dim, nb_patch, bn_mode, use_mbd, batch_size, do_plot):
    if model_name == 'generator_unet_upsampling':
        model = generator_unet_upsampling(img_dim, bn_mode, model_name=model_name)
        model.summary()
        if do_plot:
            keras.utils.plot_model(model, to_file='%s.png' % model_name, show_shapes=True, show_layer_names=True)
        return model
    if model_name == 'generator_unet_deconv':
        model = generator_unet_deconv(img_dim, bn_mode, batch_size, model_name=model_name)
        model.summary()
        if do_plot:
            keras.utils.plot_model(model, to_file='%.png' % model_name, show_layer_names=True, show_shapes=True)
        return model
    if model_name == 'DCGAN_discriminator':
        model = DCGAN_discriminator(img_dim, nb_patch, bn_mode, model_name=model_name, use_mbd=use_mbd)
        model.summary()
        if do_plot:
            keras.utils.plot_model(model, to_file='%.png' % model_name, show_shapes=True, show_layer_names=True)
        return model





