# https://github.com/leehomyc/cyclegan-1
import os
import tensorflow
import tensorflow.contrib.slim as slim
import numpy
import skimage
import skimage.transform
from imageio import imwrite
import datetime
import random
import nibabel
import glob
import matplotlib.pyplot as plt

# tensorflow.enable_eager_execution()


def cycle_consistency_loss(real_images, generate_images):
    return tensorflow.reduce_mean(tensorflow.abs(real_images - generate_images))


def lsgan_loss_generator(prob_fake_is_real):
    return tensorflow.reduce_mean(tensorflow.squared_difference(prob_fake_is_real, 1))


def lsgan_loss_discriminator(prob_real_is_real, prob_fake_is_real):
    return (tensorflow.reduce_mean(tensorflow.squared_difference(prob_real_is_real, 1)) + tensorflow.reduce_mean(
        tensorflow.squared_difference(prob_fake_is_real, 0))) * 0.5


def lrelu(x, leak=0.2, name='lrelu', alt_relu_impl=False):
    with tensorflow.variable_scope(name):
        if alt_relu_impl:
            f1 = 0.5 * (1 + leak)
            f2 = 0.5 * (1 - leak)
            return f1 * x + f2 * abs(x)
        else:
            return tensorflow.maximum(x, leak * x)


def instance_norm(x):
    with tensorflow.variable_scope('instance_norm'):
        epsilon = 1e-5
        mean, var = tensorflow.nn.moments(x, [1, 2], keep_dims=True)
        scale = tensorflow.get_variable('scale', [x.get_shape()[-1]],
                                        initializer=tensorflow.truncated_normal_initializer(mean=1.0, stddev=0.02))
        offset = tensorflow.get_variable('offset', [x.get_shape()[-1]],
                                         initializer=tensorflow.constant_initializer(0.0))
        out = scale * tensorflow.div(x - mean, tensorflow.sqrt(var + epsilon)) + offset
        return out


def general_conv2d(input_conv, o_d=64, f_h=7, f_w=7, s_h=1, s_w=1, stddev=0.02, padding='VALID', name='conv2d',
                   do_norm=True, do_relu=True, relu_factor=0):
    with tensorflow.variable_scope(name):
        conv = tensorflow.layers.conv2d(input_conv, o_d, f_w, s_w, padding, activation=None,
                                        kernel_initializer=tensorflow.truncated_normal_initializer(stddev=stddev),
                                        bias_initializer=tensorflow.constant_initializer(0.0))
        if do_norm:
            conv = instance_norm(conv)
        if do_relu:
            if relu_factor == 0:
                conv = tensorflow.nn.relu(conv, 'relu')
            else:
                conv = lrelu(conv, relu_factor, 'lrelu')
        return conv


def general_deconv2d(input_conv, out_shape, o_d=64, f_h=7, f_w=7, s_h=1, s_w=1, stddev=0.02, padding='VALID',
                     name='deconv2d', do_norm=True, do_relu=True, relu_factor=0):
    with tensorflow.variable_scope(name):
        conv = tensorflow.layers.conv2d_transpose(input_conv, o_d, [f_h, f_w], [s_h, s_w], padding, activation=None,
                                                  kernel_initializer=tensorflow.truncated_normal_initializer(
                                                      stddev=stddev),
                                                  bias_initializer=tensorflow.constant_initializer(0.0))
        if do_norm:
            conv = instance_norm(conv)
        if do_relu:
            if relu_factor == 0:
                conv = tensorflow.nn.relu(conv, 'relu')
            else:
                conv = lrelu(conv, 'lrelu')
        return conv


BATCH_SIZE = 1
IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 1
POOL_SIZE = 50
NGF = 32
NDF = 64


def build_resnet_block(inputres, dim, name='resnet', padding='REFLECT'):
    with tensorflow.variable_scope(name):
        out_res = tensorflow.pad(inputres, [[0, 0], [1, 1], [1, 1], [0, 0]], padding)
        out_res = general_conv2d(out_res, dim, 3, 3, 1, 1, 0.02, 'VALID', 'c1')
        out_res = tensorflow.pad(out_res, [[0, 0], [1, 1], [1, 1], [0, 0]], padding)
        out_res = general_conv2d(out_res, dim, 3, 3, 1, 1, 0.02, 'VALID', 'c2', do_relu=False)
        return tensorflow.nn.relu(out_res + inputres)


def build_generator_resnet_9blocks_tf(inputgen, name='generator', skip=False):
    with tensorflow.variable_scope(name):
        f = 7
        ks = 3
        padding = 'REFLECT'
        pad_input = tensorflow.pad(inputgen, [[0, 0], [ks, ks], [ks, ks], [0, 0]], padding)
        o_c1 = general_conv2d(pad_input, NGF, f, f, 1, 1, 0.02, name='c1')
        o_c2 = general_conv2d(o_c1, NGF * 2, ks, ks, 2, 2, 0.02, 'SAME', 'c2')
        o_c3 = general_conv2d(o_c2, NGF * 4, ks, ks, 2, 2, 0.02, 'SAME', 'c3')

        o_r1 = build_resnet_block(o_c3, NGF * 4, 'r1', padding)
        o_r2 = build_resnet_block(o_r1, NGF * 4, 'r2', padding)
        o_r3 = build_resnet_block(o_r2, NGF * 4, 'r3', padding)
        o_r4 = build_resnet_block(o_r3, NGF * 4, 'r4', padding)
        o_r5 = build_resnet_block(o_r4, NGF * 4, 'r5', padding)
        o_r6 = build_resnet_block(o_r5, NGF * 4, 'r6', padding)
        o_r7 = build_resnet_block(o_r6, NGF * 4, 'r7', padding)
        o_r8 = build_resnet_block(o_r7, NGF * 4, 'r8', padding)
        o_r9 = build_resnet_block(o_r8, NGF * 4, 'r9', padding)

        o_c4 = general_deconv2d(o_r9, [BATCH_SIZE, 128, 128, NGF * 2], NGF * 2, ks, ks, 2, 2, 0.02, 'SAME', 'c4')
        o_c5 = general_deconv2d(o_c4, [BATCH_SIZE, 256, 256, NGF], NGF, ks, ks, 2, 2, 0.02, 'SAME', 'c5')
        o_c6 = general_conv2d(o_c5, IMG_CHANNELS, f, f, 1, 1, 0.02, 'SAME', 'c6', do_norm=False, do_relu=False)

        if skip is True:
            out_gen = tensorflow.nn.tanh(inputgen + o_c6, 't1')
        else:
            out_gen = tensorflow.nn.tanh(o_c6, 't1')
        return out_gen


def build_generator_resnet_9blocks(inputgen, name='generator', skip=False):
    with tensorflow.variable_scope(name):
        f = 7
        ks = 3
        padding = 'CONSTANT'
        pad_input = tensorflow.pad(inputgen, [[0, 0], [ks, ks], [ks, ks], [0, 0]], padding)
        o_c1 = general_conv2d(pad_input, NGF, f, f, 1, 1, 0.02, 'c1')
        o_c2 = general_conv2d(o_c1, NGF * 2, ks, ks, 2, 2, 0.02, 'SAME', 'c2')
        o_c3 = general_conv2d(o_c2, NGF * 4, ks, ks, 2, 2, 0.02, 'SAME', 'c3')

        o_r1 = build_resnet_block(o_c3, NGF * 4, 'r1', padding)
        o_r2 = build_resnet_block(o_r1, NGF * 4, 'r2', padding)
        o_r3 = build_resnet_block(o_r2, NGF * 4, 'r3', padding)
        o_r4 = build_resnet_block(o_r3, NGF * 4, 'r4', padding)
        o_r5 = build_resnet_block(o_r4, NGF * 4, 'r5', padding)
        o_r6 = build_resnet_block(o_r5, NGF * 4, 'r6', padding)
        o_r7 = build_resnet_block(o_r6, NGF * 4, 'r7', padding)
        o_r8 = build_resnet_block(o_r7, NGF * 4, 'r8', padding)
        o_r9 = build_resnet_block(o_r8, NGF * 4, 'r9', padding)

        o_c4 = general_deconv2d(o_r9, [BATCH_SIZE, 128, 128, NGF * 2], NGF * 2, ks, ks, 2, 2, 0.02, 'SAME', 'c4')
        o_c5 = general_deconv2d(o_c4, [BATCH_SIZE, 256, 256, NGF], NGF, ks, ks, 2, 2, 0.02, 'SAME', 'c5')
        o_c6 = general_conv2d(o_c5, IMG_CHANNELS, f, f, 1, 1, 0.02, 'SAME', 'c6', do_norm=False, do_relu=False)

        if skip is True:
            out_gen = tensorflow.nn.tanh(inputgen + o_c6, 't1')
        else:
            out_gen = tensorflow.nn.tanh(o_c6, 't1')
        return out_gen


def discriminator_tf(input_disc, name='discriminator'):
    with tensorflow.variable_scope(name):
        f = 4

        o_c1 = general_conv2d(input_disc, NDF, f, f, 2, 2, 0.02, 'SAME', 'c1', do_norm=False, relu_factor=0.2)
        o_c2 = general_conv2d(o_c1, NDF * 2, f, f, 2, 2, 0.02, 'SAME', 'c2', relu_factor=0.2)
        o_c3 = general_conv2d(o_c2, NDF * 4, f, f, 2, 2, 0.02, 'SAME', 'c3', relu_factor=0.2)
        o_c4 = general_conv2d(o_c3, NDF * 8, f, f, 2, 2, 0.02, 'SAME', 'c4', relu_factor=0.2)
        o_c5 = general_conv2d(o_c4, 1, f, f, 1, 1, 0.02, 'SAME', 'c5', do_norm=False, do_relu=False)
        return o_c5


def discriminator(inputdisc, name='discriminator'):
    with tensorflow.variable_scope(name):
        f = 4
        pad_w = 2

        pad_input = tensorflow.pad(inputdisc, [[0, 0], [pad_w, pad_w], [pad_w, pad_w], [0, 0]], 'CONSTANT')
        o_c1 = general_conv2d(pad_input, NDF, f, f, 2, 2, 0.02, 'VALID', 'c1', do_norm=False, relu_factor=0.2)
        pad_o_c1 = tensorflow.pad(o_c1, [[0, 0], [pad_w, pad_w], [pad_w, pad_w], [0, 0]], 'CONSTANT')
        o_c2 = general_conv2d(pad_o_c1, NDF * 2, f, f, 2, 2, 0.02, 'VALID', 'c2', relu_factor=0.2)
        pad_o_c2 = tensorflow.pad(o_c2, [[0, 0], [pad_w, pad_w], [pad_w, pad_w], [0, 0]], 'CONSTANT')
        o_c3 = general_conv2d(pad_o_c2, NDF * 4, f, f, 2, 2, 0.02, 'VALID', 'c3', relu_factor=0.2)
        pad_o_c3 = tensorflow.pad(o_c3, [[0, 0], [pad_w, pad_w], [pad_w, pad_w], [0, 0]], 'CONSTANT')
        o_c4 = general_conv2d(pad_o_c3, NDF * 8, f, f, 1, 1, 0.02, 'VALID', 'c4', relu_factor=0.2)
        pad_o_c4 = tensorflow.pad(o_c4, [[0, 0], [pad_w, pad_w], [pad_w, pad_w], [0, 0]], 'CONSTANT')
        o_c5 = general_conv2d(pad_o_c4, 1, f, f, 1, 1, 0.02, 'VALID', 'c5', do_norm=False, do_relu=False)
        return o_c5


def patch_discriminator(inputdisc, name='discriminator'):
    with tensorflow.variable_scope(name):
        f = 4

        patch_input = tensorflow.random_crop(inputdisc, [1, 70, 70, IMG_CHANNELS])
        o_c1 = general_conv2d(patch_input, NDF, f, f, 2, 2, 0.02, 'SAME', 'c1', do_norm=False, relu_factor=0.2)
        o_c2 = general_conv2d(o_c1, NDF * 2, f, f, 2, 2, 0.02, 'SAME', 'c2', relu_factor=0.2)
        o_c3 = general_conv2d(o_c2, NDF * 4, f, f, 2, 2, 0.02, 'SAME', 'c3', relu_factor=0.2)
        o_c4 = general_conv2d(o_c3, NDF * 8, f, f, 2, 2, 0.02, 'SAME', 'c4', relu_factor=0.2)
        o_c5 = general_conv2d(o_c4, 1, f, f, 1, 1, 0.02, 'SAME', 'c5', do_norm=False, do_relu=False)
        return o_c5


def get_outputs(inputs, network='tensorflow', skip=False):
    images_a = inputs['images_a']
    images_b = inputs['images_b']
    fake_pool_a = inputs['fake_pool_a']
    fake_pool_b = inputs['fake_pool_b']

    with tensorflow.variable_scope('Model') as scope:
        if network == 'pytorch':
            current_discriminator = discriminator
            current_generator = build_generator_resnet_9blocks
        elif network == 'tensorflow':
            current_discriminator = discriminator_tf
            current_generator = build_generator_resnet_9blocks_tf
        else:
            raise ValueError('Network must be either TensorFlow oder PyTorch')

        prob_real_a_is_real = current_discriminator(images_a, 'd_A')
        prob_real_b_is_real = current_discriminator(images_b, 'd_B')

        fake_images_b = current_generator(images_a, name='g_A', skip=skip)
        fake_images_a = current_generator(images_b, name='g_B', skip=skip)

        scope.reuse_variables()

        prob_fake_a_is_real = current_discriminator(fake_images_a, 'd_A')
        prob_fake_b_is_real = current_discriminator(fake_images_b, 'd_B')

        cycle_images_a = current_generator(fake_images_b, 'g_B', skip=skip)
        cycle_images_b = current_generator(fake_images_a, 'g_A', skip=skip)

        scope.reuse_variables()

        prob_fake_pool_a_is_real = current_discriminator(fake_pool_a, 'd_A')
        prob_fake_pool_b_is_real = current_discriminator(fake_pool_b, 'd_B')

        return {
            'prob_real_a_is_real': prob_real_a_is_real,
            'prob_real_b_is_real': prob_real_b_is_real,
            'prob_fake_a_is_real': prob_fake_a_is_real,
            'prob_fake_b_is_real': prob_fake_b_is_real,
            'prob_fake_pool_a_is_real': prob_fake_pool_a_is_real,
            'prob_fake_pool_b_is_real': prob_fake_pool_b_is_real,
            'cycle_images_a': cycle_images_a,
            'cycle_images_b': cycle_images_b,
            'fake_images_a': fake_images_a,
            'fake_images_b': fake_images_b
        }


class CycleGAN:
    def __init__(self,
                 pool_size=50,
                 lambda_a=10.0,
                 lambda_b=10.0,
                 output_root_dir='cycle_tf_output',
                 to_restore=False,
                 base_lr=2e-4,
                 max_step=2,
                 network_version='tensorflow',
                 checkpoint_dir='cycle_tf_chkpts',
                 do_flipping=False,
                 skip=False):
        current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

        self._pool_size = pool_size
        self._size_before_crop = 286
        self._lambda_a = lambda_a
        self._lambda_b = lambda_b
        self._output_dir = os.path.join(output_root_dir, current_time)
        self._images_dir = os.path.join(self._output_dir, 'imgs')
        self._num_imgs_to_save = 2
        self._to_restore = to_restore
        self._base_lr = base_lr
        self._max_step = max_step
        self._network_version = network_version
        self._checkpoint_dir = checkpoint_dir
        self._do_flipping = do_flipping
        self._skip = skip

        self.fake_images_A = numpy.zeros((self._pool_size, 1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
        self.fake_images_B = numpy.zeros((self._pool_size, 1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    def model_setup(self):
        self.input_a = tensorflow.placeholder(tensorflow.float32, [1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS], name='input_A')
        self.input_b = tensorflow.placeholder(tensorflow.float32, [1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS], name='input_B')
        self.fake_pool_A = tensorflow.placeholder(tensorflow.float32, [None, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS], name='fake_pool_A')
        self.fake_pool_B = tensorflow.placeholder(tensorflow.float32, [None, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS], name='fake_pool_B')

        self.global_step = tensorflow.train.get_or_create_global_step()
        self.num_fake_inputs = 0
        self.learning_rate = tensorflow.placeholder(tensorflow.float32, shape=[], name='lr')

        inputs = {
            'images_a': self.input_a,
            'images_b': self.input_b,
            'fake_pool_a': self.fake_pool_A,
            'fake_pool_b': self.fake_pool_B
        }

        outputs = get_outputs(inputs, network=self._network_version, skip=self._skip)
        self.prob_real_a_is_real = outputs['prob_real_a_is_real']
        self.prob_real_b_is_real = outputs['prob_real_b_is_real']
        self.fake_images_a = outputs['fake_images_a']
        self.fake_images_b = outputs['fake_images_b']
        self.prob_fake_a_is_real = outputs['prob_fake_a_is_real']
        self.prob_fake_b_is_real = outputs['prob_fake_b_is_real']
        self.cycle_images_a = outputs['cycle_images_a']
        self.cycle_images_b = outputs['cycle_images_b']
        self.prob_fake_pool_a_is_real = outputs['prob_fake_pool_a_is_real']
        self.prob_fake_pool_b_is_real = outputs['prob_fake_pool_b_is_real']

    def compute_losses(self):
        cycle_consistency_loss_a = self._lambda_a * cycle_consistency_loss(
            real_images=self.input_a,
            generate_images=self.cycle_images_a
        )
        cycle_consistency_loss_b = self._lambda_b * cycle_consistency_loss(
            real_images=self.input_a,
            generate_images=self.cycle_images_b
        )

        lsgan_loss_a = lsgan_loss_generator(self.prob_fake_a_is_real)
        lsgan_loss_b = lsgan_loss_generator(self.prob_fake_b_is_real)

        g_loss_A = cycle_consistency_loss_a + cycle_consistency_loss_b + lsgan_loss_b
        g_loss_B = cycle_consistency_loss_b + cycle_consistency_loss_a + lsgan_loss_a

        d_loss_A = lsgan_loss_discriminator(
            prob_real_is_real=self.prob_real_a_is_real,
            prob_fake_is_real=self.prob_fake_a_is_real
        )
        d_loss_B = lsgan_loss_discriminator(
            prob_real_is_real=self.prob_real_b_is_real,
            prob_fake_is_real=self.prob_fake_b_is_real
        )

        optimizer = tensorflow.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5)
        self.model_vars = tensorflow.trainable_variables()

        d_A_vars = [var for var in self.model_vars if 'd_A' in var.name]
        d_B_vars = [var for var in self.model_vars if 'd_B' in var.name]
        g_A_vars = [var for var in self.model_vars if 'g_A' in var.name]
        g_B_vars = [var for var in self.model_vars if 'g_B' in var.name]

        for var in self.model_vars:
            print('var:', var.name)

        self.d_A_trainer = optimizer.minimize(d_loss_A, var_list=d_A_vars)
        self.d_B_trainer = optimizer.minimize(d_loss_B, var_list=d_B_vars)
        self.g_A_trainer = optimizer.minimize(g_loss_A, var_list=g_A_vars)
        self.g_B_trainer = optimizer.minimize(g_loss_B, var_list=g_B_vars)

        # Summaries for TensorBoard
        self.g_A_loss_summ = tensorflow.summary.scalar('g_A_loss', g_loss_A)
        self.g_B_loss_summ = tensorflow.summary.scalar('g_B_loss', g_loss_B)
        self.d_A_loss_summ = tensorflow.summary.scalar('d_A_loss', d_loss_A)
        self.d_B_loss_summ = tensorflow.summary.scalar('d_B_loss', d_loss_B)

    def fake_image_pool(self, num_fakes, fake, fake_pool):
        if num_fakes < self._pool_size:
            fake_pool[num_fakes] = fake
            return fake
        else:
            p = random.random()
            if p > 0.5:
                random_id = random.randint(0, self._pool_size - 1)
                temp = fake_pool[random_id]
                fake_pool[random_id] = fake
                return temp
            else:
                return fake

    def save_images(self, sess, epoch):
        if not os.path.exists(self._images_dir):
            os.makedirs(self._images_dir)
        names = ['inputA_', 'inputB_', 'fakeA_', 'fakeB_', 'cycA_', 'cycB_']

        with open(os.path.join(self._output_dir, 'epoch_' + str(epoch) + '.html'), 'w') as v_html:
            for i in range(0, self._num_imgs_to_save):
                print('Saving images {}/{}'.format(i, self._num_imgs_to_save))
                inputs = sess.run(self.inputs)
                fake_A_temp, fake_B_temp, cyc_A_temp, cyc_B_temp = sess.run([
                    self.fake_images_a,
                    self.fake_images_b,
                    self.cycle_images_a,
                    self.fake_images_a
                ], feed_dict={
                    self.input_a: inputs['images_i'],
                    self.input_b: inputs['images_j']
                })

                tensors = [inputs['images_i'], inputs['images_j'], fake_B_temp, fake_A_temp, cyc_A_temp, cyc_B_temp]

                for name, tensor in zip(names, tensors):
                    image_name = name + str(epoch) + '_' + str(i) + '.jpg'
                    imwrite(os.path.join(self._images_dir, image_name), ((tensor[0] + 1) * 127.5).astype(numpy.uint8))
                    v_html.write('<img src=\"' + os.path.join('imgs', image_name) + '\">')
                    v_html.write('<br>')

    def load_data(self, size_before_crop=286, do_shuffle=False, do_flipping=False):
        data_path = 'D:\Halimeh\Datasets\MSD\Task05_Prostate\imagesTr'
        files = glob.glob(data_path + '\p*')
        print('file_names', files)
        image_i = []
        image_j = []
        # i = 0
        for file_path in files:
            # i += 1
            # if i >= 2:
                # break

            nifti = nibabel.load(file_path)
            slices = nifti.get_data()
            print('slices', type(slices), slices.shape)
            print('slices', numpy.min(slices), numpy.max(slices))
            scaling, intercept, mn, mx = nibabel.volumeutils.calculate_scale(slices, numpy.uint8,allow_intercept=True)
            slices = (slices - intercept) / scaling
            slices[slices > 255] = 255
            slices[slices < 0] = 0
            slices = (slices - 127.5) / 127.5
            # slices.astype(int)
            print('slices', numpy.min(slices), numpy.max(slices))
            slices = skimage.transform.resize(slices, output_shape=(IMG_HEIGHT, IMG_WIDTH, slices.shape[2], slices.shape[3]))
            for v in range(slices.shape[2]):
                mi = slices[..., v, 0]
                mi = mi[:, :, None]
                mj = slices[..., v, 1]
                mj = mj[:, :, None]
                image_i.append(mi)
                image_j.append(mj)
        image_i = numpy.stack(image_i)
        image_j = numpy.stack(image_j)
        print('image_i', image_i.shape, 'image_j', image_j.shape)
        '''
        image_i = tensorflow.image.resize_images(slices[..., [v], 0], [size_before_crop, size_before_crop])
        image_j = tensorflow.image.resize_images(slices[..., [v], 1], [size_before_crop, size_before_crop])
        if do_flipping:
            image_i = tensorflow.image.random_flip_left_right(image_i)
            image_j = tensorflow.image.random_flip_left_right(image_j)

        image_i = tensorflow.random_crop(image_i, [IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS])
        image_j = tensorflow.random_crop(image_j, [IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS])
        image_i = tensorflow.subtract(tensorflow.div(image_i, 127.5), 1)
        image_j = tensorflow.subtract(tensorflow.div(image_j, 127.5), 1)
        '''

        # Batch
        if do_shuffle:
            images_i, images_j = tensorflow.train.shuffle_batch([image_i, image_j], 1, 5000, 100, enqueue_many=True)
        else:
            images_i, images_j = tensorflow.train.batch([image_i, image_j], 1, enqueue_many=True)
        print('batch', images_i.get_shape())
        inputs = {
            'images_i': images_i,
            'images_j': images_j
        }
        return inputs

    def train(self):
        print('Training is on')

        self.inputs = self.load_data()
        print('self.inputs', type(self.inputs), type(self.inputs['images_i']), type(self.inputs['images_j']))




        self.model_setup()
        self.compute_losses()

        init = (tensorflow.global_variables_initializer(), tensorflow.local_variables_initializer())
        saver = tensorflow.train.Saver()
        max_images = 602

        with tensorflow.Session() as sess:
            sess.run(init)
            # u = sess.run(self.inputs)


            if self._to_restore:
                chkpt_fname = tensorflow.train.latest_checkpoint(self._checkpoint_dir)
                saver.restore(sess, chkpt_fname)

            writer = tensorflow.summary.FileWriter(self._output_dir)
            if not os.path.exists(self._output_dir):
                os.makedirs(self._output_dir)

            coordinator = tensorflow.train.Coordinator()
            threads = tensorflow.train.start_queue_runners(coord=coordinator)

            # Training Loop
            for epoch in range(sess.run(self.global_step), self._max_step):
                print('Epoch', epoch)
                saver.save(sess, os.path.join(self._output_dir, 'cyclegan'), global_step=epoch)

                if epoch < 100:
                    curr_lr = self._base_lr
                else:
                    curr_lr = self._base_lr - self._base_lr * (epoch - 100) / 100

                self.save_images(sess, epoch)

                for i in range(0, max_images):
                    print('Processing batch {}/{}'.format(i, max_images))

                    inputs = sess.run(self.inputs)
                    '''
                    print('PLOTTING S')
                    fig, ax = plt.subplots(1, 2)
                    ax[0].imshow(((inputs['images_i'][0, ..., 0] + 1) * 127.5).astype(numpy.uint8), cmap='gray')
                    ax[0].axis('off')
                    ax[1].imshow(((inputs['images_j'][0, ..., 0] + 1) * 127.5).astype(numpy.uint8), cmap='gray')
                    ax[1].axis('off')
                    plt.show()
                    print('PLOTTING E')
                    '''
                    '''
                    imwrite(os.path.join(self._images_dir, 'images_i.jpg'),
                            ((inputs['images_i'][0] + 1) * 127.5).astype(numpy.uint8))
                    imwrite(os.path.join(self._images_dir, 'images_j.jpg'),
                            ((inputs['images_j'][0] + 1) * 127.5).astype(numpy.uint8))
                    '''

                    # Optimize the G_A network
                    _, fake_B_temp, summary_str = sess.run(
                        [self.g_A_trainer,
                         self.fake_images_b,
                         self.g_A_loss_summ],
                        feed_dict={
                            self.input_a: inputs['images_i'],
                            self.input_b: inputs['images_j'],
                            self.learning_rate: curr_lr
                        }
                    )
                    writer.add_summary(summary_str, epoch * max_images + i)

                    fake_B_temp1 = self.fake_image_pool(self.num_fake_inputs, fake_B_temp, self.fake_images_B)

                    # Optimize the D_B network
                    _, summary_str = sess.run(
                        [self.d_B_trainer, self.d_B_loss_summ],
                        feed_dict={
                            self.input_a: inputs['images_i'],
                            self.input_b: inputs['images_j'],
                            self.learning_rate: curr_lr,
                            self.fake_pool_B: fake_B_temp1
                        }
                    )
                    writer.add_summary(summary_str, epoch * max_images + i)

                    # Optimize the G_B network
                    _, fake_A_temp, summary_str = sess.run(
                        [self.g_B_trainer,
                         self.fake_images_a,
                         self.g_B_loss_summ],
                        feed_dict={
                            self.input_a: inputs['images_i'],
                            self.input_b: inputs['images_j'],
                            self.learning_rate: curr_lr
                        }
                    )
                    writer.add_summary(summary_str, epoch * max_images + i)

                    fake_A_temp1 = self.fake_image_pool(self.num_fake_inputs, fake_A_temp, self.fake_images_A)

                    # Optimize the D_A network
                    _, summary_str = sess.run(
                        [self.d_A_trainer, self.d_A_loss_summ],
                        feed_dict={
                            self.input_a: inputs['images_i'],
                            self.input_b: inputs['images_j'],
                            self.learning_rate: curr_lr,
                            self.fake_pool_A: fake_A_temp1
                        }
                    )
                    writer.add_summary(summary_str, epoch * max_images + i)

                    writer.flush()
                    self.num_fake_inputs += 1
                sess.run(tensorflow.assign(self.global_step, epoch + 1))

            coordinator.request_stop()
            coordinator.join(threads)
            writer.add_graph(sess.graph)

    def test(self):
        print('Testing the results')

        self.inputs = None

        self.model_setup()
        saver = tensorflow.train.Saver()
        init = tensorflow.global_variables_initializer()

        with tensorflow.Session() as sess:
            sess.run(init)
            chkpt_fname = tensorflow.train.latest_checkpoint(self._checkpoint_dir)
            saver.restore(sess, chkpt_fname)

            coord = tensorflow.train.Coordinator()
            threads = tensorflow.train.start_queue_runners(coord=coord)

            coord.request_stop()
            coord.join(threads)


cycle_gan_model = CycleGAN()
cycle_gan_model.train()
