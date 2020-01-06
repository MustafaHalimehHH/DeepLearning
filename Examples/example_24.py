# https://machinelearningmastery.com/how-to-develop-a-pix2pix-gan-for-image-to-image-translation/
import os
import numpy
import keras
import matplotlib.pyplot as plt



def load_images(path, size=(256, 512)):
    src_list, tar_list = list(), list()
    for filename in os.listdir(path):
        pixels = keras.preprocessing.image.load_img(path + filename, target_size=size)
        pixels = keras.preprocessing.image.img_to_array(pixels)
        sat_img, map_img = pixels[:, :256], pixels[:, 256:]
        src_list.append(sat_img)
        tar_list.append(map_img)
    return [numpy.asarray(src_list), numpy.asarray(tar_list)]


def define_discriminator(image_shape):
    init = keras.initializers.RandomNormal(stddev=2e-2)
    in_src_img = keras.layers.Input(shape=image_shape)
    in_target_img = keras.layers.Input(shape=image_shape)
    merged = keras.layers.Concatenate()([in_src_img, in_target_img])
    d = keras.layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(merged)
    d = keras.layers.LeakyReLU(alpha=2e-1)(d)
    d = keras.layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = keras.layers.BatchNormalization()(d)
    d = keras.layers.LeakyReLU(alpha=2e-1)(d)
    d = keras.layers.Conv2D(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = keras.layers.BatchNormalization()(d)
    d = keras.layers.LeakyReLU(alpha=2e-1)(d)
    d = keras.layers.Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = keras.layers.BatchNormalization()(d)
    d = keras.layers.LeakyReLU(alpha=2e-1)(d)
    d = keras.layers.Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = keras.layers.BatchNormalization()(d)
    d = keras.layers.LeakyReLU(alpha=2e-1)(d)
    d = keras.layers.Conv2D(1, (4, 4), padding='same', kernel_initializer=init)(d)
    patch_out = keras.layers.Activation('sigmoid')(d)
    model = keras.models.Model([in_src_img, in_target_img], patch_out)
    opt = keras.optimizers.Adam(lr=2e-4, beta_1=5e-1)
    model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
    return model


def define_encoder_block(layer_in, n_filters, batchnorm=True):
    init = keras.initializers.RandomNormal(stddev=2e-2)
    g = keras.layers.Conv2D(n_filters, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(layer_in)
    if batchnorm:
        g = keras.layers.BatchNormalization()(g, training=True)
    g = keras.layers.LeakyReLU(alpha=2e-1)(g)
    return g


def define_decoder_block(layer_in, skip_in, n_filters, dropout=True):
    init = keras.initializers.RandomNormal(stddev=2e-2)
    g = keras.layers.Conv2DTranspose(n_filters, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(layer_in)
    g = keras.layers.BatchNormalization()(g, training=True)
    if dropout:
        g = keras.layers.Dropout(5e-1)(g, training=True)
    g = keras.layers.Concatenate()([g, skip_in])
    g = keras.layers.Activation('relu')(g)
    return g


def define_generator(image_shape=(256, 256, 3)):
    init = keras.initializers.RandomNormal(stddev=2e-2)
    in_image = keras.layers.Input(shape=image_shape)
    e1 = define_encoder_block(in_image, 64, batchnorm=False)
    e2 = define_encoder_block(e1, 128)
    e3 = define_encoder_block(e2, 256)
    e4 = define_encoder_block(e3, 512)
    e5 = define_encoder_block(e4, 512)
    e6 = define_encoder_block(e5, 512)
    e7 = define_encoder_block(e6, 512)

    b = keras.layers.Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(e7)
    b = keras.layers.Activation('relu')(b)
    d1 = define_decoder_block(b, e7, 512)
    d2 = define_decoder_block(d1, e6, 512)
    d3 = define_decoder_block(d2, e5, 512)
    d4 = define_decoder_block(d3, e4, 512, dropout=False)
    d5 = define_decoder_block(d4, e3, 256, dropout=False)
    d6 = define_decoder_block(d5, e2, 128, dropout=False)
    d7 = define_decoder_block(d6, e1, 64, dropout=False)

    g = keras.layers.Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d7)
    out_image = keras.layers.Activation('tanh')(g)
    model = keras.models.Model(in_image, out_image)
    return model


def define_gan(g_model, d_model, image_shape):
    d_model.trainable = False
    in_src = keras.layers.Input(shape=image_shape)
    gen_out = g_model(in_src)
    dis_out = d_model([in_src, gen_out])
    model = keras.models.Model(in_src, [dis_out, gen_out])
    opt = keras.optimizers.Adam(lr=2e-4, beta_1=5e-1)
    model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1, 100])
    return model


def load_real_samples(file_name):
    data = numpy.load(file_name)
    X1, X2 = data['arr_0'], data['arr_1']
    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5
    return [X1, X2]


def generate_real_samples(dataset, n_samples, patch_shape):
    train_A, train_B = dataset
    ix = numpy.random.randint(0, train_A.shape[0], n_samples)
    X1, X2 = train_A[ix], train_B[ix]
    y = numpy.ones((n_samples, patch_shape, patch_shape, 1))
    return [X1, X2], y


def generate_fake_samples(g_model, samples, patch_shape):
    X = g_model.predict(samples)
    y = numpy.ones((len(X), patch_shape, patch_shape, 1))
    return X, y


def summarize(step, g_model, dataset, n_sampels=3):
    [X_real_A, X_real_B], _ = generate_real_samples(dataset, n_sampels, 1)
    X_fake_B, _ = generate_fake_samples(g_model, X_real_A, 1)
    X_real_A = (X_real_A + 1.0) / 2.0
    X_real_B = (X_real_B + 1.0) / 2.0
    X_fake_B = (X_fake_B + 1.0) / 2.0
    for i in range(n_sampels):
        plt.subplot(3, n_sampels, 1 + i)
        plt.axis('off')
        plt.imshow(X_real_A[i])
    for i in range(n_sampels):
        plt.subplot(3, n_sampels, 1 + n_sampels + i)
        plt.axis('off')
        plt.imshow(X_real_B[i])
    for i in range(n_sampels):
        plt.subplot(3, n_sampels, 1 + n_sampels * 2 + i)
        plt.axis('off')
        plt.imshow(X_fake_B[i])
    plt.show()



def train(d_model, g_model, gan_model, dataset, n_epochs=2, n_batch=1):
    n_patch = d_model.output_shape[1]
    train_A, train_B = dataset
    bat_per_epo = int(len(train_A) / n_batch)
    n_steps = bat_per_epo * n_epochs
    print('bat_per_epo', bat_per_epo)
    print('n_steps', n_steps)

    for i in range(n_steps):
        [X_real_A, X_real_B], y_real = generate_real_samples(dataset, n_batch, n_patch)
        X_fake_B, y_fake = generate_fake_samples(g_model, X_real_A, n_patch)
        d_loss1 = d_model.train_on_batch([X_real_A, X_real_B], y_real)
        d_loss2 = d_model.train_on_batch([X_real_A, X_fake_B], y_fake)
        g_loss, _, _ = gan_model.train_on_batch(X_real_A, [y_real, X_real_B])
        print('>%d, d1[%.3f], d2[%.3f], g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss))
        if (i+1) % (bat_per_epo / 2) == 0:
            summarize(i, g_model, dataset)



# [src_images, tar_images] = load_images('C:\\Users\\halimeh\\Downloads\\maps.tar\\maps\\train\\')
# print('Loaded', src_images.shape, tar_images.shape)
# numpy.savez_compressed('maps_256.npz', src_images, tar_images)


dataset = load_real_samples('maps_256.npz')
dataset = [dataset[0][:100], dataset[1][:100]]
print('dataset', dataset[0].shape, dataset[1].shape)
image_shape = dataset[0].shape[1:]
print('image_shape', image_shape)
d_model = define_discriminator(image_shape)
g_model = define_generator(image_shape)
gan_model = define_gan(g_model, d_model, image_shape)
train(d_model, g_model, gan_model, dataset)
