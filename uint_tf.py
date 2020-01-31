# https://github.com/taki0112/UNIT-Tensorflow/
import tensorflow
import tensorflow.contrib as contrib


def batch_norm(x, is_training=False, scope='batch_norm'):
    return contrib.layers.batch_norm(x, decay=0.9, epsilon=1e-05, center=True, scale=True, updates_collections=None, is_training=is_training, scope=scope)


def instance_norm(x, scope='instance'):
    return contrib.layers.instance_norm(x, epsilon=1e-05, center=True, scale=True, scope=scope)


def activation(x, activation_fn='leaky'):
    assert activation_fn in ['leaky', 'relu', 'tanh', 'sigmoid', 'swish', None], 'No valid activation_fn'

    if activation_fn == 'leaky':
        x = tensorflow.nn.leaky_relu(x, alpha=1e-2)
    if activation_fn == 'relu':
        x = tensorflow.nn.relu(x)
    if activation_fn == 'sigmoid':
        x = tensorflow.nn.sigmoid(x)
    if activation_fn == 'tanh':
        x = tensorflow.nn.tanh(x)
    if activation_fn == 'swish':
        x = x * tensorflow.nn.sigmoid(x)

    return x


def conv(x, channels, kernel=3, stride=2, pad=0, normal_weight_init=False, activation_fn='leaky', scope='conv_0'):
    with tensorflow.variable_scope(scope):
        x = tensorflow.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
        if normal_weight_init:
            x = tensorflow.layers.conv2d(inputs=x, filters=channels, kernel_size=kernel,
                                         kernel_initializer=tensorflow.truncated_normal_initializer(stddev=2e-2),
                                         strides=stride, kernel_regularizer=contrib.layers.l2_regularizer(scale=1e-4))
        else:
            if activation_fn == 'relu':
                x = tensorflow.layers.conv2d(inputs=x, filters=channels, kernel_size=kernel,
                                             kernel_initializer=contrib.layers.variance_scaling_initializer(),
                                             strides=stride,
                                             kernel_regularizer=contrib.layers.l2_regularizer(scale=1e-4))
            else:
                x = tensorflow.layers.conv2d(inputs=x, filters=channels, kernel_size=kernel, strides=stride,
                                             kernel_regularizer=contrib.layers.l2_regularizer(scale=1e-4))

        x = activation(x, activation_fn)
        return x


def deconv(x, channels, kernel=3, stride=2, normal_weight_init=False, activation_fn='leaky', scope='deconv_0'):
    with tensorflow.variable_scope(scope):
        if normal_weight_init:
            x = tensorflow.layers.conv2d_transpose(inputs=x, filters=channels, kernel_size=kernel, kernel_initializer=tensorflow.truncated_normal_initializer(stddev=2e-2),
                                                   strides=stride, padding='SAME', kernel_regularizer=contrib.layers.l2_regularizer(scale=1e-4))
        else:
            if activation_fn == 'relu':
                x = tensorflow.layers.conv2d_transpose(inputs=x, filters=channels, kernel_size=kernel, kernel_initializer=contrib.layers.variance_scaling_initializer(),
                                                       strides=stride, kernel_regularizer=contrib.layers.l2_regularizer(scale=1e-4))
            else:
                x = tensorflow.layers.conv2d_transpose(inputs=x, filters=channels, kernel_size=kernel, strides='SAME', kernel_regularizer=contrib.layers.l2_regularizer(scale=1e-4))
        x = activation(x, activation_fn)
        return x


def res_block(x_init, channels, kernel=3, stride=1, pad=1, dropout_ratio=0.0, normal_weight_init=False, is_training=True, norm_fn='instance', scope='resblock_0'):
    assert norm_fn in ['instance', 'batch', 'weight', 'spectral', None], 'No valid norm_fn'

    with tensorflow.variable_scope(scope):
        with tensorflow.variable_scope('res_1'):
            x = tensorflow.pad(x_init, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
            if normal_weight_init:
                x = tensorflow.layers.conv2d(inputs=x, filters=channels, kernel_size=kernel, kernel_initializer=tensorflow.truncated_normal_initializer(stddev=2e-2),
                                             strides=stride, kernel_regularizer=contrib.layers.l2_regularizer(scale=1e-04))
            else:
                x = tensorflow.layers.conv2d(inputs=x, filters=channels, kernel_size=kernel, kernel_initializer=contrib.layers.variance_scaling_initializer(),
                                             strides=stride, kernel_regularizer=contrib.layers.l2_regularizer(scale=1e-04))
            if norm_fn == 'instance':
                x = instance_norm(x, 'res_1_instance')
            if norm_fn == 'batch':
                x = batch_norm(x, is_training, 'res_1_batch')
            x = tensorflow.nn.relu(x)

        with tensorflow.variable_scope('res_2'):
            x = tensorflow.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
            if normal_weight_init:
                x = tensorflow.layers.conv2d(inputs=x, filters=channels, kernel_size=kernel, kernel_initializer=tensorflow.truncated_normal_initializer(stddev=2e-02),
                                             strides=stride, kernel_regularizer=contrib.layers.l2_regularizer(scale=1e-04))
            else:
                x = tensorflow.layers.conv2d(inputs=x, filters=channels, kernel_size=kernel, kernel_initializer=contrib.layers.variance_scaling_initializer(),
                                             strides=stride, kernel_regularizer=contrib.layers.l2_regularizer(scale=1e-04))
            if norm_fn == 'instance':
                x = instance_norm(x, 'res_2_instance')
            if norm_fn == 'batch':
                x = batch_norm(x, 'res_2_batch')
            if dropout_ratio > 0.0:
                x = tensorflow.layers.dropout(x, rate=dropout_ratio, training=is_training)
            return x + x_init


def gaussian_noise_layer(mu):
    sigma = 1.0
    gaussian_random_vector = tensorflow.random_normal(shape=tensorflow.shape(mu), mean=0.0, stddev=1.0, dtype=tensorflow.float32)
    return mu + sigma * gaussian_random_vector


def KL_divergence(mu):
    mu_2 = tensorflow.square(mu)
    loss = tensorflow.reduce_mean(mu_2)
    return loss


def L1_loss(x, y):
    loss = tensorflow.reduce_mean(tensorflow.abs(x - y))
    return loss


def discriminator_loss(real, fake, smoothing=False, use_lsgan=False):
    if use_lsgan:
        if smoothing:
            real_loss = tensorflow.reduce_mean(tensorflow.squared_difference(real, 0.9)) * 0.5
        else:
            real_loss = tensorflow.reduce_mean(tensorflow.squared_difference(real, 1.0)) * 0.5
        fake_loss = tensorflow.reduce_mean(tensorflow.square(fake)) * 0.5
    else:
        if smoothing:
            real_labels = tensorflow.fill(tensorflow.shape(real), 0.9)
        else:
            real_labels = tensorflow.ones_like(real)
        fake_labels = tensorflow.zeros_like(fake)

        real_loss = tensorflow.reduce_mean(tensorflow.nn.sigmoid_cross_entropy_with_logits(labels=real_labels, logits=real))
        fake_loss = tensorflow.reduce_mean(tensorflow.nn.sigmoid_cross_entropy_with_logits(labels=fake_labels, logits=fake))

    loss = real_loss + fake_loss
    return loss


def generator_loss(fake, smoothing=False, use_lsgan=False):
    if use_lsgan:
        if smoothing:
            loss = tensorflow.reduce_mean(tensorflow.squared_difference(fake, 0.9)) * 0.5
        else:
            loss = tensorflow.reduce_mean(tensorflow.squared_difference(fake, 1.0)) * 0.5
    else:
        if smoothing:
            fake_labels = tensorflow.fill(tensorflow.shape(fake), 0.9)
        else:
            fake_labels = tensorflow.ones_like(fake)
        loss = tensorflow.reduce_mean(tensorflow.nn.sigmoid_cross_entropy_with_logits(labels=fake_labels, logits=fake))
    return loss


class UNIT(object):
    def __init__(self, sess):
        self.model_name = 'UNIT'
        self.sess = sess
        self.checkpoint_dir = 'UNIT_output'
        self.result_dir = 'UNIT_output'
        self.log_dir = 'UNIT_output'
        self.sample_dir = 'UNIT_output'

        self.epoch = 200
        self.batch_size = 1
        self.lr = 1e-04
        self.KL_weight = 1e-01
        self.L1_weight = 100.0
        self.KL_cycle_weight = 1e-01
        self.L1_cycle_weight = 100.0
        self.GAN_weight = 10.0

        # Encoder
        self.ch = 64
        self.n_encoder = 3
        self.n_enc_res_block = 3
        self.n_enc_share = 1

        # Generator
        self.n_gen_share = 1
        self.n_gen_res_block = 3
        self.n_gen_decoder = 3

        # Discriminator
        self.n_dis = 6

        self.res_dropout = 0.0
        self.smoothing = False
        self.lsgan = False
        self.norm = 'instance'
        self.replay_memory = False
        self.pool_size = 50
        self.img_size = 256
        self.img_ch = 1
        self.augment_flag = True
        self.augment_size = self.img_size + (30 if self.img_size == 256 else 15)
        self.normal_weight_init = True

    def encoder(self, x, is_training=True, reuse=False, scope='encoder'):
        channel = self.ch
        with tensorflow.variable_scope(scope, reuse=reuse):
            x = conv(x, channel, kernel=7, stride=1, pad=3, normal_weight_init=self.normal_weight_init, activation_fn='leaky', scope='conv_0')
            for i in range(1, self.n_encoder):
                x = conv(x, channel * 2, kernel=3, stride=2, pad=1, normal_weight_init=self.normal_weight_init, activation_fn='leaky', scope='conv' + str(i))
                channel *= 2
            for i in range(0, self.n_enc_res_block):
                x = res_block(x, channel, kernel=3, stride=1, pad=1, dropout_ratio=self.res_dropout, normal_weight_init=self.normal_weight_init, is_training=is_training, norm_fn=self.norm, scope='res_block' + str(i))
            return x

    def share_encoder(self, x, is_training=True, reuse=False, scope='share_encoder'):
        channel = self.ch * pow(2, self.n_encoder - 1)
        with tensorflow.variable_scope(scope, reuse=reuse):
            for i in range(0, self.n_enc_share):
                x = res_block(x, channel, kernel=3, stride=1, pad=1, dropout_ratio=self.res_dropout, normal_weight_init=self.normal_weight_init,
                              is_training=is_training, norm_fn=self.norm, scope='res_block' + str(i))
            x = gaussian_noise_layer(x)
            return x

    def share_generator(self, x, is_training=True, reuse=False, scope='share_generator'):
        channel = self.ch * pow(2, self.n_encoder - 1)
        with tensorflow.varaible_scope(scope, reuse=reuse):
            for i in range(0, self.n_gen_share):
                x = res_block(x, channel, kernel=3, stride=1, pad=1, dropout_ratio=self.res_dropout, normal_weight_init=self.normal_weight_init, is_training=is_training, norm_fn=self.norm, scope='res_block' + str(i))
            return x

    def generator(self, x, is_training=True, reuse=False, scope='generator'):
        channel = self.ch * pow(2, self.n_encoder - 1)
        with tensorflow.variable_scope(scope, reuse=reuse):
            for i in range(0, self.n_gen_res_block):
                x = res_block(x, channel, kernel=3, stride=1, pad=1, dropout_ratio=self.res_dropout, normal_weight_init=self.normal_weight_init, is_training=is_training, norm_fn=self.norm, scope='res_block' + str(i))
            for i in range(0, self.n_gen_decoder):
                x = deconv(x, channel // 2, kernel=3, stride=2, normal_weight_init=self.normal_weight_init, activation_fn='leaky', scope='deconv_' + str(i))
                channel = channel // 2
            x = deconv(x, self.img_ch, kernel=1, stride=1, normal_weight_init=self.normal_weight_init, activation_fn='tanh', scope='deconv_tanh')
            return x

    def discriminator(self, x, reuse=False, scope='discriminator'):
        channel = self.ch
        with tensorflow.variable_scope(scope, reuse=reuse):
            x = conv(x, channel, kernel=3, stride=2, pad=1, normal_weight_init=self.normal_weight_init, activation_fn='leaky', scope='conv_0')
            for i in range(1, self.n_dis):
                x = conv(x, channel * 2, kernel=3, stride=2, pad=1, normal_weight_init=self.normal_weight_init, activation_fn='leaky', scope='conv_' + str(i))
                channel *= 2
            x = conv(x, channels=1, kernel=1, stride=1, pad=0, normal_weight_init=self.normal_weight_init, activation_fn=None, scope='dis_logit')
            return x

    def translation(self, x_A, x_B):
        





























