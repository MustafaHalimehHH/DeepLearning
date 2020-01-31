# https://github.com/tjwei/GANotebooks/blob/master/CycleGAN-keras.ipynb
import os
import keras
import keras.backend as k
import numpy
import glob
import nibabel
import NNKeras.nifti_utils
import matplotlib.pyplot as plt

os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['THEANO_FLAGS'] = 'floatX=float32, device=cuda, optimizer=fast_run, dnn.library_path=/usr/lib'
if os.environ['KERAS_BACKEND'] == 'theano':
    channel_axis = 1
    k.set_image_data_format('channels_first')
    channel_first = True
else:
    k.set_image_data_format('channels_last')
    channel_axis = -1
    channel_first = False


def __conv_init(a):
    print('conv_init', a)
    k = keras.initializers.RandomNormal(0, 2e-2)(a)
    k.conv_weight = True
    return k


conv_init = keras.initializers.RandomNormal(0, 2e-2)
gamma_init = keras.initializers.RandomNormal(1., 2e-2)  # Batch Normalization

if k._BACKEND == 'theano':
    import keras.backend.theano_backend as theano_backend


    def _preprocess_conv2d_kernel(kernel, data_format):
        if hasattr(kernel, 'original'):
            print('use original')
            return kernel.original
        elif hasattr(kernel, '_keras_shape'):
            s = kernel._keras_shape
            print('use reshape', s)
            kernel = kernel.reshape((s[3], s[2], s[0], s[1]))
        else:
            kernel = kernel.dimshuffle((3, 3, 0, 1))
        return kernel


    theano_backend._preprocess_conv2d_kernel = _preprocess_conv2d_kernel


def conv_2d(f, *a, **k):
    return keras.layers.Conv2D(filters=f, kernel_initializer=conv_init, *a, **k)


def batch_norm():
    return keras.layers.BatchNormalization(momentum=0.9, axis=channel_axis, epsilon=1.01e-5,
                                           gamma_initializer=gamma_init)


def BASIC_D(nc_in, ndf, max_layers=3, use_sigmoid=True):
    '''
    DCGAN_D(nc, ndf, max_layers=3)
    nc: channels
    ndf: filters of the first layer
    max_layers: max hidden layers
    '''
    if channel_first:
        input_a = keras.layers.Input(shape=(nc_in, None, None))
    else:
        input_a = keras.layers.Input(shape=(None, None, nc_in))
    _ = input_a
    _ = conv_2d(ndf, kernel_size=4, strides=2, padding='same', name='First')(_)
    _ = keras.layers.LeakyReLU(alpha=0.2)(_)

    for layer in range(1, max_layers):
        out_feat = ndf * min(2 ** layer, 8)
        _ = conv_2d(out_feat, kernel_size=4, strides=2, padding='same', use_bias=False,
                    name='pyramid.{0}'.format(layer))(_)
        _ = batch_norm()(_, training=1)
        _ = keras.layers.LeakyReLU(alpha=0.2)(_)
    out_feat = ndf * min(2 ** max_layers, 8)
    _ = keras.layers.ZeroPadding2D(1)(_)
    _ = conv_2d(out_feat, kernel_size=4, use_bias=False, name='pyramid_last')(_)
    _ = batch_norm()(_, training=1)
    _ = keras.layers.LeakyReLU(alpha=0.2)(_)

    # final_layer
    _ = keras.layers.ZeroPadding2D(1)(_)
    _ = conv_2d(1, kernel_size=4, name='final'.format(out_feat, 1), activation='sigmoid' if use_sigmoid else None)(_)
    return keras.Model(inputs=[input_a], outputs=_)


def UNET_G(isize, nc_in=3, nc_out=3, ngf=64, fixed_input_size=True):
    max_nf = 8 * ngf

    def block(x, s, nf_in, use_batchnorm=True, nf_out=None, nf_next=None):
        assert s >= 2 and s % 2 == 0, 'Assertion test failed'
        if nf_next is None:
            nf_next = min(nf_in * 2, max_nf)
        if nf_out is None:
            nf_out = nf_in
        x = conv_2d(nf_next, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s > 2)), padding='same',
                    name='conv_{0}'.format(s))(x)
        if s > 2:
            if use_batchnorm:
                x = batch_norm()(x, training=1)
            x2 = keras.layers.LeakyReLU(alpha=0.2)(x)
            x2 = block(x2, s // 2, nf_next)
            x = keras.layers.Concatenate(axis=channel_axis)([x, x2])
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Conv2DTranspose(nf_out, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                                         kernel_initializer=conv_init, name='conv_trans.{0}'.format(s))(x)
        x = keras.layers.Cropping2D(1)(x)
        if use_batchnorm:
            x = batch_norm()(x, training=1)
        if s <= 8:
            x = keras.layers.Dropout(0.5)(x, training=1)
        return x

    s = isize if fixed_input_size else None
    if channel_first:
        _ = inputs = keras.layers.Input(shape=(nc_in, s, s))
    else:
        _ = inputs = keras.layers.Input(shape=(s, s, nc_in))
    _ = block(_, isize, nc_in, False, nf_out=nc_out, nf_next=ngf)
    _ = keras.layers.Activation('tanh')(_)
    return keras.Model(inputs=inputs, outputs=[_])


nc_in = 1  # 3
nc_out = 1  # 3
ngf = 64
ndf = 64
use_lsgan = True
λ = 10 if use_lsgan else 100
print('λ', type(λ), λ)
load_size = 143
image_size = 128
batch_size = 1
lrD = 2e-4
lrG = 2e-4

netDA = BASIC_D(nc_in, ndf, use_sigmoid=not use_lsgan)
netDB = BASIC_D(nc_out, ndf, use_sigmoid=not use_lsgan)
netDA.summary()
netDB.summary()

netGA = UNET_G(image_size, nc_in, nc_out, ngf)
netGB = UNET_G(image_size, nc_out, nc_in, ngf)
netGA.summary()
netGB.summary()

if use_lsgan:
    loss_fn = lambda output, target: k.mean(k.abs(k.square(output - target)))
else:
    loss_fn = lambda output, target: -k.mean(k.log(output + 1e-12) * target + k.log(1 - output + 1e-12) * (1 - target))


def cycle_variables(netG1, netG2):
    real_input = netG1.inputs[0]
    fake_output = netG1.outputs[0]
    rec_input = netG2([fake_output])
    fn_generate = k.function([real_input], [fake_output, rec_input])
    return real_input, fake_output, rec_input, fn_generate


real_A, fake_B, rec_A, cycleA_generate = cycle_variables(netGB, netGA)
real_B, fake_A, rec_B, cycleB_generate = cycle_variables(netGA, netGB)


def D_loss(netD, real, fake, rec):
    output_real = netD([real])
    output_fake = netD([fake])
    loss_D_real = loss_fn(output_real, k.ones_like(output_real))
    loss_D_fake = loss_fn(output_fake, k.zeros_like(output_fake))
    loss_G = loss_fn(output_fake, k.ones_like(output_fake))
    loss_D = loss_D_real + loss_D_fake
    loss_cyc = k.mean(k.abs(rec - real))
    return loss_D, loss_G, loss_cyc


loss_DA, loss_GA, loss_cycA = D_loss(netDA, real_A, fake_A, rec_A)
loss_DB, loss_GB, loss_cycB = D_loss(netDB, real_B, fake_B, rec_B)
loss_cyc = loss_cycA + loss_cycB
loss_G = loss_GA, loss_GB + λ * loss_cyc
loss_D = loss_DA + loss_DB

weightsD = netDA.trainable_weights + netDB.trainable_weights
weightsG = netGA.trainable_weights + netGB.trainable_weights

training_updates = keras.optimizers.Adam(lr=lrD, beta_1=0.5).get_updates(weightsD, [], loss_D)
netD_train = k.function([real_A, real_B], [loss_DA / 2, loss_DB / 2], training_updates)
training_updates = keras.optimizers.Adam(lr=lrG, beta_1=0.5).get_updates(weightsG, [], loss_G)
netG_train = k.function([real_A, real_B], [loss_GA, loss_GB, loss_cyc], training_updates)


def load_data():
    '''
    dataset_path = 'maps_256.npz'
    dataset = numpy.load(dataset_path)
    A, B = dataset['arr_0'], dataset['arr_1']
    A = (A - 127.5) / 127.5
    B = (B - 127.5) / 127.5
    return A, B
    '''
    data_path = 'D:\Halimeh\Datasets\MSD\Task05_Prostate\imagesTr'
    modal_1 = []
    modal_2 = []
    files = glob.glob(data_path + '\p*')
    print('file_names', files)
    for file_path in files:
        nifti = nibabel.load(file_path)
        slices = nifti.get_data()
        print('slices', slices.shape)

        resized = NNKeras.nifti_utils.resize(nifti, 256, 256)
        resized = NNKeras.nifti_utils.scale_to_grayscale(resized)
        print('resized', resized.min(), resized.max())
        for i in range(resized.shape[2]):
            m_1 = resized[..., i, 0]
            m_1 = m_1[:, :, None]
            modal_1.append(m_1)
            m_2 = resized[..., i, 1]
            m_2 = m_2[:, :, None]
            modal_2.append(m_2)

    print('modal_1', len(modal_1), modal_1[0].shape)
    print('modal_2', len(modal_2), modal_2[0].shape)
    modal_1 = numpy.stack(modal_1)
    modal_2 = numpy.stack(modal_2)
    print(modal_1.shape, modal_2.shape, modal_1.ndim, modal_2.ndim)
    numpy.random.shuffle(modal_1)
    numpy.random.shuffle(modal_2)
    modal_1 = (modal_1 - 127.5) / 127.5
    modal_2 = (modal_2 - 127.5) / 127.5
    return modal_1, modal_2


def mini_batch_AB(A, B):
    print('A', A.shape, 'B', B.shape)
    i = 0
    while True:
        print('i', i)
        if i >= A.shape[0]:
            i = 0
        yield A[[i], ...], B[[i], ...]
        i += 1
    '''
    for i in range(A.shape[0]):
        print('i', i, A[[i], ...].shape, B[[i], ...].shape)
        yield A[[i], ...], B[[i], ...]
    '''
    '''
    while True:
        i = numpy.random.randint(0, A.shape[0], size=1)
        print('i', i, A[i, ...].shape, B[i, ...].shape)
        fig, ax = plt.subplots(1, 2)
        a = 0.5 * A[i][0] + 0.5
        b = 0.5 * B[i][0] + 0.5
        ax[0].imshow(a)
        ax[0].axis('off')
        ax[1].imshow(b)
        ax[1].axis('off')
        plt.show()
        yield A[i, ...], B[i, ...]
        '''


def show_G(a, b):
    assert a.shape == b.shape, 'Assertion shapes Equality failed'

    def G(fn_generator, x):
        r = numpy.array([fn_generator([x[i: i + 1]]) for i in range(x.shape[0])])
        return r.swapaxes(0, 1)[:, :, 0]

    r_a = G(cycleA_generate, a)
    r_b = G(cycleB_generate, b)
    print('a', a.shape, 'b', b.shape, 'r_a', r_a.shape, 'r_b', r_b.shape)
    a = 0.5 * a + 0.5
    b = 0.5 * b + 0.5
    r_a = 0.5 * r_a + 0.5
    r_b = 0.5 * r_b + 0.5

    arr = numpy.concatenate([a, b, r_a[0], r_b[0], r_a[1], r_b[1]])
    print('arr', arr.shape)

    fig, ax = plt.subplots(3, 2)
    cnt = 0
    for i in range(3):
        for j in range(2):
            # ax[i, j].imshow(arr[cnt])
            ax[i, j].imshow(arr[cnt, :, :, 0], cmap='gray')
            ax[i, j].axis('off')
            cnt += 1
    plt.show()


A, B = load_data()
niter = 150
gen_iterations = 0
epoch = 0
err_CYC_sum = err_GA_sum = err_GB_sum = err_DA_sum = err_DB_sum = 0
display_iters = 2

train_batch = mini_batch_AB(A, B)
while epoch < niter:
    gen_iterations = 0
    while gen_iterations < A.shape[0]:
        a, b = next(train_batch)
        err_DA, err_DB = netD_train([a, b])
        err_DA_sum += err_DA
        err_DB_sum += err_DB

        err_GA, err_GB, err_CYC = netG_train([a, b])
        err_GA_sum += err_GA
        err_GB_sum += err_GB
        err_CYC_sum += err_CYC
        gen_iterations += 1

        if gen_iterations % display_iters == 0:
            print('[%d/%d/%d] Loss_D: %f %f Loss_G: %f %f Loss_CYC: %f' % (
            niter, epoch, gen_iterations, err_DA_sum / display_iters, err_DB_sum / display_iters,
            err_GA_sum / display_iters, err_GB_sum / display_iters, err_CYC_sum / display_iters))
        if gen_iterations % 50 == 0:
            show_G(a, b)

    epoch += 1
