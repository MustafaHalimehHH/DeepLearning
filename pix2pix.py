# https://github.com/tdeboissiere/DeepLearningImplementations/blob/master/pix2pix/src/model/models.py

import time
import keras
import keras.backend as k
import numpy


def l1_loss(y_true, y_pred):
    return k.sum(k.abs(y_pred - y_true), axis=-1)


def get_nb_patch(img_dim, patch_size, image_data_format):
    assert image_data_format in ['channels_first', 'channels_last'], 'Bad image formation'
    if image_data_format == 'channels_first':
        assert img_dim[1] % patch_size[0] == 0, 'patch_size_height does not divide image_height'
        assert img_dim[2] % patch_size[1] == 0, 'patch_size_width does not divide image_width'
        nb_patch = (img_dim[1] // patch_size[0]) * (img_dim[2] * patch_size[1])
        img_dim_disc = (img_dim[0], patch_size[0], patch_size[1])
    else:
        assert img_dim[0] % patch_size[0] == 0, 'patch_size_height does not divide image_height'
        assert img_dim[1] % patch_size[1] == 0, 'patch_size_width does not divide image_width'
        nb_patch = (img_dim[0] // patch_size[0]) * (img_dim[1] // patch_size[1])
        img_dim_disc = (patch_size[0], patch_size[1], img_dim[-1])
    return nb_patch, img_dim_disc


def extract_patches(x, image_data_format, patch_size):
    if image_data_format == 'channels_first':
        x = x.transpose(0, 2, 3, 1)

    list_X = []
    list_row_idx = [(i * patch_size[0], (i + 1) * patch_size[0]) for i in range(x.shape[1] // patch_size[0])]
    list_col_idx = [(i * patch_size[1], (i + 1) * patch_size[1]) for i in range(x.shape[2] // patch_size[1])]
    for row_idx in list_row_idx:
        for col_idx in list_col_idx:
            list_X.append(x[:, row_idx[0]: row_idx[1], col_idx[0]: col_idx[1], :])
    if image_data_format == 'channels_first':
        for i in range(len(list_X)):
            list_X[i] = list_X.transpose(0, 3, 1, 2)
    return list_X


def gen_batch(x1, x2, batch_size):
    while True:
        idx = numpy.random.choice(x1.shape[0], batch_size, replace=False)
        yield x1[idx], x2[idx]


def get_disc_batch(x_full_batch, x_sketch_batch, generator_model, batch_counter, patch_size, image_data_format, label_smoothing=False, label_flipping=0):
    if batch_counter % 2 == 0:
        x_disc = generator_model.predcit(x_sketch_batch)
        y_disc = numpy.zeros((x_disc.shape[0], 2), dtype=numpy.uint8)
        y_disc[:, 0] = 1
        if label_flipping > 0:
            p = numpy.random.binomial(1, label_flipping)
            if p > 0:
                y_disc[:, [0, 1]] = y_disc[:, [1, 0]]
    else:
        x_disc = x_full_batch
        y_disc = numpy.zeros((x_disc.shape[0], 2), dtype=numpy.uint8)
        if label_smoothing:
            y_disc[:, 1] = numpy.random.uniform(low=0.9, high=1, size=y_disc.shape[0])
        else:
            y_disc[:, 1] = 1
        if label_flipping > 0:
            p = numpy.random.binomial(1, label_flipping)
            if p > 0:
                y_disc[:, [0, 1]] = y_disc[:, [1, 0]]
    x_disc = extract_patches(x_disc, image_data_format, patch_size)
    return x_disc, y_disc


def minb_disc(x):
    diffs = k.expand_dims(x, 3) - k.expand_dims(k.permute_dimensions(x, [1, 2, 0]), 0)
    abs_diffs = k.sum(k.abs(diffs), 2)
    x = k.sum(k.exp(-abs_diffs), 2)
    return x


def lambda_output(input_shape):
    return input_shape[:2]


def conv_block_unet(x, f, name, bn_axis, bn=True, strides=(2, 2)):
    x = keras.layers.LeakyReLU(0.2)(x)
    x = keras.layers.Conv2D(f, (3, 3), strides=strides, name=name, padding='same')(x)
    if bn:
        x = keras.layers.BatchNormalization(axis=bn_axis)(x)
    return x


def up_conv_block_unet(x, x2, name, bn_axis, bn=True, dropout=False):
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.UpSampling2D(size=(2, 2))(x)
    x = keras.layers.Conv2D(x, (3, 3), name=name, padding='same')(x)
    if bn:
        x = keras.layers.BatchNormalization(axis=bn_axis)(x)
    if dropout:
        x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Concatenate(axis=bn_axis)([x, x2])
    return x


def deconv_block_unet(x, x2, f, h, w, batch_size, name, bn_axis, bn=True, dropout=False):
    output_shape = (batch_size, h * 2, w * 2, f)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Deconv2D(f, (3, 3), output_shape=output_shape, strides=(2, 2), padding='same')(x)
    if bn:
        x = keras.layers.BatchNormalization(axis=bn_axis)(x)
    if dropout:
        x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Concatenate(axis=bn_axis)([x, x2])
    return x


def generator_unet_upsamling(img_dim, model_name='generator_unet_upsampling'):
    nb_filters = 64
    if k.image_dim_ordering() == 'channels_first':
        bn_axis = 1
        nb_channels = img_dim[0]
        min_s = min(img_dim[1:])
    else:
        bn_axis = -1
        nb_channels = img_dim[-1]
        min_s = min(img_dim[:-1])
    print('bn_axis, nb_channels, min_s', bn_axis, nb_channels, min_s)

    unet_input = keras.layers.Input(shape=img_dim, name='unet_input')

    nb_conv = int(numpy.floor(numpy.log(min_s) / numpy.log(2)))
    print('nb_conv', nb_conv)
    list_nb_filters = [nb_filters * min(8, 2 ** i) for i in range(nb_conv)]
    print('Encoder: list_nb_filters', list_nb_filters)
    list_encoder = [
        keras.layers.Conv2D(list_nb_filters[0], (3, 3), strides=(2, 2), padding='same', name='unet_conv2D_1')(
            unet_input)]
    for i, f in enumerate(list_nb_filters[1:]):
        name = 'unet_conv2D_%s' % (i + 2)
        conv = conv_block_unet(list_encoder[-1], f, name, bn_axis)
        list_encoder.append(conv)

    list_nb_filters = list_nb_filters[:-2][::-1]
    print('Decoder list_nb_filters', list_nb_filters)
    if len(list_nb_filters < nb_conv - 1):
        list_nb_filters.append(nb_filters)

    list_decoder = [
        up_conv_block_unet(list_encoder[-1], list_encoder[-2], list_nb_filters[0], 'unet_upconv2D_1', bn_axis,
                           dropout=True)]
    for i, f in enumerate(list_nb_filters[1:]):
        name = 'unet_upconv2D_%s' % (i + 2)
        if i < 2:
            d = True
        else:
            d = False
        conv = up_conv_block_unet(list_decoder[-1], list_encoder[-(i + 3)], f, name, bn_axis, dropout=d)
        list_decoder.append(conv)

    x = keras.layers.Activation('relu')(list_decoder[-1])
    x = keras.layers.UpSampling2D(size=(2, 2))(x)
    x = keras.layers.Conv2D(nb_channels, (3, 3), name='last_conv', padding='same')(x)
    x = keras.layers.Activation('tanh')(x)
    generator_unet = keras.models.Model(inputs=[unet_input], outputs=[x])
    return generator_unet


def generator_unet_deconv(img_dim, batch_size, model_name='generator_unet_deconv'):
    assert k.backend() == 'tensorflow', 'Not working on others'

    nb_filters = 64
    bn_axis = -1
    h, w, nb_channels = img_dim
    min_s = min(img_dim[:-1])

    unet_input = keras.layers.Input(shape=img_dim, name='unet_input')
    nb_conv = int(numpy.floor(numpy.log(min_s) / numpy.log(2)))
    list_nb_filtes = [nb_filters * min(8, 2 ** i) for i in range(nb_conv)]
    list_encoder = [
        keras.layers.Conv2D(list_nb_filtes[0], (3, 3), strides=(2, 2), name='unet_conv2D_1', padding='same')(
            unet_input)]
    h, w = h / 2, w / 2
    for i, f in enumerate(list_nb_filtes[1:]):
        name = 'unet_conv2D_%s' % (i + 2)
        conv = conv_block_unet(list_encoder[-1], f, name, bn_axis)
        list_encoder.append(conv)
        h, w = h / 2, w / 2

    list_nb_filtes = list_nb_filtes[:-1][::-1]
    if len(list_nb_filtes) < nb_conv - 1:
        list_nb_filtes.append(nb_filters)

    list_decoder = [
        deconv_block_unet(list_encoder[-1], list_encoder[-2], list_nb_filtes[0], h, w, batch_size, 'unet_upconv2D_1',
                          bn_axis, dropout=True)]
    h, w = h * 2, w * 2
    for i, f in enumerate(list_nb_filtes[1:]):
        name = 'unet_upconv2D_%s' % (i + 2)
        if i < 2:
            d = True
        else:
            d = False
        conv = deconv_block_unet(list_decoder[-1], list_encoder[-(i + 3)], f, h, w, batch_size, name, bn_axis,
                                 dropout=d)
        list_decoder.append(conv)
        h, w = h * 2, w * 2

    x = keras.layers.Activation('relu')(list_decoder[-1])
    output_shape = (batch_size, ) + img_dim
    x = keras.layers.Deconv2D(nb_channels, (3, 3), output_shape=output_shape, strides=(2, 2), padding='same')(x)
    x = keras.layers.Activation('tanh')(x)
    generator_unet = keras.models.Model(inputs=[unet_input], outputs=[x])
    return generator_unet


def DCGAN_discriminator(img_dim, nb_patch, model_name='DCGAN_discriminator', use_mbd=True):
    list_input = [keras.layers.Input(shape=img_dim, name='disc_input_%s' % i) for i in range(nb_patch)]
    if k.image_dim_ordering() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = -1

    nb_filters = 64
    nb_conv = int(numpy.floor(numpy.log(img_dim[1])/ numpy.log(2)))
    list_filters = [nb_filters * min(8, 2 ** i) for i in nb_conv]
    x_input = keras.layers.Input(shape=img_dim, name='discriminator_input')
    x = keras.layers.Conv2D(list_filters[0], (3, 3), strides=(2, 2), name='disc_conv2d_1', padding='same')(x_input)
    x = keras.layers.BatchNormalization(axis=bn_axis)(x)
    x = keras.layers.LeakyReLU(0.2)(x)

    for i, f in enumerate(list_filters[1:]):
        name = 'disc_conv2d_%s' % (i + 2)
        x = keras.layers.Conv2D(f, (3, 3), strides=(2, 2), name=name, padding='same')(x)
        x = keras.layers.BatchNormalization(axis=bn_axis)(x)
        x = keras.layers.LeakyReLU(0.2)(x)

    x_flat = keras.layers.Flatten()(x)
    x = keras.layers.Dense(2, activation='softmax', name='disc_dense')(x_flat)
    PatchGAN = keras.models.Model(inputs=[x_input], outputs=[x, x_flat])
    print('PatchGAN summary')
    PatchGAN.summary()

    x = [PatchGAN(patch)[0] for patch in list_input]
    x_mbd = [PatchGAN(patch)[1] for patch in list_input]

    if len(x) > 1:
        x = keras.layers.Concatenate(x)
    else:
        x = x[0]

    if use_mbd:
        if len(x_mbd) > 1:
            x_mbd = keras.layers.Concatenate(x_mbd)
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
            if image_dim_ordering == 'channels_first':
                x_patch = keras.layers.Lambda(lambda z: z[:, row_idx[0]: row_idx[1], col_idx[0]: col_idx[1], :])(generated_image)
            else:
                x_patch = keras.layers.Lambda(lambda z: z[:, :, row_idx[0]: row_idx[1], col_idx[0]: col_idx[1]])(generated_image)
            list_gen_patch.append(x_patch)

    DCGAN_output = discriminator_model(list_gen_patch)
    DCGAN = keras.models.Model(inputs=[gen_input], outputs=[generated_image, DCGAN_output], name='DCGAN')
    return DCGAN


def load(model_name, img_dim, nb_patch, use_mbd, batch_size):
    if model_name == 'generator_unet_upsampling':
        model = generator_unet_upsamling(img_dim, model_name=model_name)
        model.summary()
        keras.utils.plot_model(model, to_file='generator_unet_upsampling.png', show_shapes=True, show_layer_names=True)
    if model_name == 'generator_unet_deconv':
        model = generator_unet_deconv(img_dim, batch_size, model_name=model_name)
        model.summary()
        keras.utils.plot_model(model, to_file='generator_unet_deconv.png', show_shapes=True, show_layer_names=True)
    if model_name == 'DCGAN_discriminator':
        model = DCGAN_discriminator(img_dim, nb_patch, model_name=model_name, use_mbd=use_mbd)
        model.summary()
        keras.utils.plot_model(model, to_file='DCGAN_discriminator.png', show_layer_names=True, show_shapes=True)


def train():
    image_data_format = 'channels_last'
    img_dim = (224, 224, 1)
    batch_size = 32
    nb_epoch = 10
    patch_size = 28
    n_batch_per_epoch = 12
    use_mbd = True
    nb_patch, img_dim_disc = get_nb_patch(img_dim, patch_size, image_data_format=image_data_format)

    x_full_train, x_sketch_train, x_full_val, x_sketch_val = None, None, None, None

    generator_model = generator_unet_deconv(img_dim=img_dim, batch_size=batch_size)
    generator_model.compile(loss='mae', optimizer=keras.optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-8))

    discriminator_model = DCGAN_discriminator(img_dim=img_dim, nb_patch=nb_patch, use_mbd=use_mbd)
    discriminator_model.trainable = False

    DCGAN_model = DCGAN(generator_model, discriminator_model, img_dim, patch_size, image_data_format)
    DCGAN_model.compile(loss=[l1_loss, 'binary_crossentropy'], loss_weights=[1E1, 1], optimizer=keras.optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-8))

    discriminator_model.trainable = True
    discriminator_model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-8))

    gen_loss = 100
    disc_loss = 100
    print('Start Training')
    for e in range(nb_epoch):
        batch_counter = 1
        start = time.time()
        for x_full_batch, x_sketch_batch in gen_batch(x_full_train, x_sketch_train, batch_size):
            x_disc, y_disc = get_disc_batch(x_full_batch, x_sketch_batch, generator_model, batch_counter, patch_size, image_data_format)
            disc_loss = discriminator_model.train_on_batch(x_disc, y_disc)
            x_gen_target, x_gen = next(gen_batch(x_full_train, x_sketch_train, batch_size))
            y_gen = numpy.zeros((x_gen.shape[0], 2), dtype=numpy.uint8)
            y_gen[:, 1] = 1

            discriminator_model.trainable = False
            gen_loss = DCGAN_model.train_on_batch(x_gen, [x_gen_target, y_gen])
            discriminator_model.trainable = True
            batch_counter += 1

            if batch_counter > n_batch_per_epoch:
                break
        print('Epoch %s/%s, Time: %s' % (e + 1, nb_epoch, time.time() - str))


