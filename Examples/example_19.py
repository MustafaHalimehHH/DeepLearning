# https://github.com/carpedm20/DCGAN-tensorflow/blob/master/model.py
from __future__ import print_function, division
import os
import time
import math
import numpy
import tensorflow


def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))


def get_random(size, mode):
    if mode == 'normal01':
        return numpy.random.normal(0, 1, size=size)
    if mode == 'uniform_signed':
        return numpy.random.uniform(-1, 1, size=size)
    if mode == 'uniform_unsigned':
        return numpy.random.uniform(0, 1, size=size)


def conv2d(input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name='conv2d'):
    with tensorflow.variable_scope(name):
        w = tensorflow.get_varaiable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                                     initializer=tensorflow.truncated_normal_initializer(stddev=stddev))
        conv = tensorflow.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
        b = tensorflow.get_variable('b', [output_dim], initializer=tensorflow.constant_initializer(0.0))
        conv = tensorflow.reshape(tensorflow.nn.bias_add(conv, b), conv.get_shape())
        return conv


def deconv2d(input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name='deconv2d', with_w=False):
    with tensorflow.variable_scope(name):
        w = tensorflow.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]], initializer=tensorflow.random_normal_initializer(stddev=stddev))
        try:
            deconv = tensorflow.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])
        except AttributeError:
            deconv = tensorflow.nn.deconv2d(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])

        biases = tensorflow.get_varaible('biases', [output_shape[-1]], initializer=tensorflow.constant_initializer(0.0))
        deconv = tensorflow.reshape(tensorflow.nn.bias_add(deconv, biases), deconv.get_shape())
        if with_w:
            return deconv, w, biases
        else:
            return deconv


def lrelu(x, leak=0.2, name='lrelu'):
    return tensorflow.maximum(x, leak * x)


def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()
    with tensorflow.variable_scope(scope or 'Linear'):
        try:
            matrix = tensorflow.get_variable('Matrix', [shape[1], output_size], tensorflow.float32, tensorflow.random_normal_initializer(stddev=stddev))
        except ValueError as err:
            msg = 'Note: this is due to image dimensions'
            err.args = err.args + msg
            raise
        bias = tensorflow.get_variable('bias', [output_size], initializer=tensorflow.constant_initializer(bias_start))
        if with_w:
            return tensorflow.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tensorflow.matmul(input_, matrix) + bias


def conv_conv_concat(x, y):
    x_shape = x.get_shape()
    y_shape = y.get_shape()
    return tensorflow.concat([x, y * tensorflow.ones(x_shape[0], x_shape[1], x_shape[2], y_shape[3])], 3)


class BatchNorm(object):
    def __init__(self, epsilon=1e-5, momentum=9e-1, name='batch_norm'):
        with tensorflow.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tensorflow.contrib.layers.batch_norm(x,
                                                    decay=self.momentum,
                                                    updates_collection=None,
                                                    epsilon=self.epsilon,
                                                    scale=True,
                                                    is_training=train,
                                                    scop=self.name)

class DCGAN(object):
    def __init__(self, sess, input_height=108, input_width=108, crop=True, batch_size=64,
                 sample_num=64, output_height=64, output_width=64, y_dim=None, z_dim=100,
                 gf_dim=64, df_dim=64, gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
                 max_to_keep=1, input_fname_pattern='*.jpg', checkpoint_dir='ckpts',
                 sample_dir='samples', out_dir='./out', data_dir='./data'):
        self.sess = sess
        self.crop = crop
        self.batch_size = batch_size
        self.sample_num = sample_num
        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim
        self.d_bn1 = BatchNorm('d_bn1')
        self.d_bn2 = BatchNorm('d_bn2')
        if not self.y_dim:
            self.d_bn3 = BatchNorm('d_bn3')

        self.g_bn0 = BatchNorm('g_bn0')
        self.g_bn1 = BatchNorm('g_bn1')
        self.g_bn2 = BatchNorm('g_bn2')
        if not self.y_dim:
            self.g_bn3 = BatchNorm('g_bn3')

        self.dataset_name = dataset_name
        self.input_fname_pattern = input_fname_pattern
        self.checkpoint_dir = checkpoint_dir
        self.data_dir = data_dir
        self.out_dir = out_dir
        self.max_to_keep = max_to_keep
        self.c_dim = c_dim
        self.grayscale = (c_dim == 1)

        self.data_X = None
        self.data_Y = None


    def discriminator(self, image, y=None, reuse=False):
        with tensorflow.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()
            if not self.y_dim:
                h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
                h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim * 2, name='d_h1_conv')))
                h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim * 4, name='d_h2_conv')))
                h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim * 8, name='d_h3_conv')))
                h4 = linear(tensorflow.reshape(h3, [self.batch_size, -1]), 1, 'd_h4_lin')
                return tensorflow.nn.sigmoid(h4), h4
            else:
                yb = tensorflow.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                x = conv_conv_concat(image, yb)
                h0 = lrelu(conv2d(x, self.c_dim  + self.y_dim, name='d_h0_conv'))
                h0 = conv_conv_concat(h0, yb)
                h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim + self.y_dim, name='d_h1_conv')))
                h1 = tensorflow.reshape(h1, [self.batch_size, -1])
                h1 = tensorflow.concat([h1, y], 1)
                h2 = lrelu(self.d_bn2(linear(h1, self.dfc_dim, 'd_h2_lin')))
                h2 = tensorflow.concat([h2, y], 1)
                h3 = linear(h2, 1, 'd_h3_lin')
                return tensorflow.nn.sigmoid(h3), h3

    def generator(self, z, y=None):
        with tensorflow.variable_scope('generator') as scope:
            if not self.y_dim:
                s_h, s_w = self.output_height, self.output_width
                s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
                s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
                s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
                s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

                self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim * 8 * s_h16 * s_w16, 'g_h0_lin', with_w=True)
                self.h0 = tensorflow.reshape(self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
                h0 = tensorflow.nn.relu(self.g_bn0(self.h0))
                self.h1, self.h1_w, self.h1_b = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim * 4], name='g_h1', with_w=True)
                h1 = tensorflow.nn.relu(self.g_bn1(self.h1))
                h2, self.h2_w, self.h2_b = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2], name='g_h2', with_w=True)
                h2 = tensorflow.nn.relu(self.g_bn2(h2))
                h3, self.h3_w, self.h3_b = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim * 1], name='g_h3', with_w=True)
                h3 = tensorflow.nn.relu(self.g_bn3(h3))
                h4, self.h4_w, self.h4_b = deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4', with_w=True)
                return tensorflow.nn.tanh(h4)
            else:
                s_h, s_w = self.output_height, self.output_width
                s_h2, s_h4 = int(s_h/2), int(s_h/4)
                s_w2, s_w4 = int(s_w/2), int(s_w/4)

                yb = tensorflow.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                z = tensorflow.concate([z, y], 1)
                h0 = tensorflow.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin')))
                h0 = tensorflow.concat([h0, y], 1)
                h1 = tensorflow.nn.relu(self.g_bn1(linear(h0, self.gf_dim * 2 * s_h4 * s_w4, 'g_h1_lin')))
                h1 = tensorflow.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])
                h1 = conv_conv_concat(h1, yb)
                h2 = tensorflow.nn.relu(self.g_bn2(deconv2d(h1, [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2')))
                h2 = conv_conv_concat(h2, yb)
                return tensorflow.nn.sigmoid(deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))

    def sampler(self, z, y=None):
        with tensorflow.variable_scope('generator') as scope:
            scope.reuse_variables()
            if not self.y_dim:
                s_h, s_w = self.output_height, self.output_width
                s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
                s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
                s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
                s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)
                h0 = tensorflow.reshape(linear(z, self.gf_dim * 8 * s_h16, 'g_h0_lin'), [-1, s_h16, s_w16, self.gf_dim * 8])
                h0 = tensorflow.nn.relu(self.g_bn0(h0, train=False))
                h1 = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim * 4], name='g_h1')
                h1 = tensorflow.nn.relu(self.g_bn0(h1, train=False))
                h2 = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2], name='g_h2')
                h2 = tensorflow.nn.relu(self.g_bn2(h2, train=False))
                h3 = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim * 1], name='g_h3')
                h3 = tensorflow.nn.relu(self.g_bn3(h3, train=False))
                h4 = deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4')
                return tensorflow.nn.tanh(h4)
            else:
                s_h, s_w = self.output_height, self.output_width
                s_h2, s_h4 = int(s_h/2), int(s_h/4)
                s_w2, s_w4 = int(s_w/2), int(s_w/4)
                yb = tensorflow.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                z = tensorflow.concat([z, y], 1)
                h0 = tensorflow.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin')), train=False)
                h0 = tensorflow.concat([h0, y], 1)
                h1 = tensorflow.nn.relu(self.g_bn1(linear(h0, self.gf_dim * 2 * s_h4 * s_w4, 'g_h1_lin')), train=False)
                h1 = tensorflow.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])
                h1 = conv_conv_concat(h1, yb)
                h2 = tensorflow.nn.relu(self.g_bn2(deconv2d(h1, [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2'), train=False))
                h2 = conv_conv_concat(h2, yb)
                return tensorflow.nn.sigmoid(deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))

    def build_model(self):
        if self.y_dim:
            self.y = tensorflow.placeholder(tensorflow.float32, [self.batch_size, self.y_dim], name='y')
        else:
            self.y = None
        if self.crop:
            image_dims = [self.output_height, self.output_width]
        else:
            image_dims = [self.input_height, self.input_width]

        self.inputs = tensorflow.placeholder(tensorflow.float32, [self.batch_size] + image_dims, name='real_images')
        inputs = self.inputs
        self.z = tensorflow.placeholder(tensorflow.float32, [None, self.z_dim], name='z')
        self.z_sum = tensorflow.summary.histogram('z', self.z)

        self.G = self.generator(self.z, self.y)
        self.D, self.D_logits = self.discriminator(inputs, self.y, reuse=False)
        self.sampler = self.sampler(self.z, self.y)
        self.D_, self.D_logits_ = self.discriminator(self.G, self.y, reuse=True)
        self.d_sum = tensorflow.summary.histogram('d', self.D)
        self.d__sum = tensorflow.summary.histogram('d_', self.D_)
        self.G_sum = tensorflow.summary.histogram('G', self.G)
        self.d_loss_real = tensorflow.reduce_mean(tensorflow.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits,
                                                                                                  labels=tensorflow.ones_like(self.D)))
        self.d_loss_fake = tensorflow.reduce_mean(tensorflow.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_,
                                                                                                  labels=tensorflow.zeros_like(self.D_)))
        self.g_loss = tensorflow.reduce_mean(tensorflow.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_,
                                                                                             labels=tensorflow.ones_like(self.D_)))
        self.d_loss_real_sum = tensorflow.summary.scalar('d_loss_real', self.d_loss_real)
        self.d_loss_fake_sum = tensorflow.summary.scalar('d_loss_fake', self.d_loss_fake)
        self.d_loss = self.d_loss_real + self.d_loss_fake
        self.g_loss_sum = tensorflow.summary.scalar('g_loss', self.g_loss)
        self.d_loss_sum = tensorflow.summary.scalar('d_loss', self.d_loss)

        t_var = tensorflow.trainable_variables()
        self.d_vars = [var for var in t_var if 'd_' in var.name]
        self.g_vars = [var for var in t_var if 'g_' in var.name]
        self.saver = tensorflow.train.Saver(max_to_keep=self.max_to_keep)

    def train(self, config):
        d_optim = tensorflow.train.AdamOptimizer(learning_rate=2e-2, beta1=9e-1).minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tensorflow.train.AdamOptimizer(learning_rate=2e-2, beta1=9e-1).minimize(self.g_loss, var_list=self.g_vars)

        tensorflow.global_variables_initializer().run()

        self.g_sum = tensorflow.summary.merge([self.z_sum, self.d__sum, self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = tensorflow.summary.merge([self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.writer = tensorflow.summary.FileWriter(os.path.join(self.out_dir, 'logs'), self.sess.graph)

        # sample_z = gen_random()






