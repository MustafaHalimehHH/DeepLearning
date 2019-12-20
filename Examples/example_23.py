# https://github.com/zhangqianhui/Conditional-GAN
import tensorflow
from tensorflow.contrib.layers.python.layers import variance_scaling_initializer, batch_norm
from tensorflow.contrib.layers.python.layers import xavier_initializer
import numpy
import keras
import cv2
import skimage
import skimage.io
import matplotlib.pyplot as plt



def lrelu(x, alpha=2e-1):
    return tensorflow.maximum(x, alpha * x)

def conv2d(input_, output_dim, k_h=3, k_w=3, d_h=2, d_w=2, name='conv2d'):
    with tensorflow.variable_scope(name):
        w = tensorflow.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim], initializer=variance_scaling_initializer())
        conv = tensorflow.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
        b = tensorflow.get_variable('b', [output_dim], initializer=tensorflow.constant_initializer(0.0))
        conv = tensorflow.reshape(tensorflow.nn.bias_add(conv, b), conv.get_shape())
        return conv, w


def de_conv2d(input_, output_shape, k_h=3, k_w=3, d_h=2, d_w=2, stddev=2e-2, name='deconv2d', with_w=False, initializer=variance_scaling_initializer()):
    with tensorflow.variable_scope(name):
        w = tensorflow.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]], initializer=initializer)
        deconv = tensorflow.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])
        b = tensorflow.get_variable('b', [output_shape[-1]], initializer=tensorflow.constant_initializer(0.0))
        deconv = tensorflow.reshape(tensorflow.nn.bias_add(deconv, b), deconv.get_shape())
        if with_w:
            return deconv, w, b
        else:
            return deconv


def fully_connected(input_, output_size, scope=None, with_w=False, initializer=variance_scaling_initializer()):
    shape = input_.get_shape().as_list()
    with tensorflow.variable_scope(scope or 'Linear'):
        matrix = tensorflow.get_variable('Matrix', [shape[1], output_size], tensorflow.float32, initializer=initializer)
        b = tensorflow.get_variable('b', [output_size], initializer=tensorflow.constant_initializer(0.0))
        if with_w:
            return tensorflow.matmul(input_, matrix) + b, matrix, b
        else:
            return tensorflow.matmul(input_, matrix) + b


def conv_cond_concat(x, y):
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tensorflow.concat([x, y * tensorflow.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)


def batch_normal(input_, scope='scope', reuse=False):
    return batch_norm(input_, epsilon=1e-5, decay=9e-1, scale=True, scope=scope, reuse=reuse, updates_collections=None)


def sample_label():
    num = 64
    label_vector = numpy.zeros((num, 10), dtype=numpy.float32)
    for i in range(0, num):
        label_vector[i, int(i/8)] = 1.0
    return label_vector


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = numpy.zeros(h * size[0], w * size[1], 3)
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h: j*h + h, i*w: i*w + w, :] = image
    return img


def save_image(images, size, path):
    skimage.io.imsave(path, merge(images, size))


def inverse_transform(image):
    return (image + 1.0) / 2.0


def save_images(images, size, image_path):
    return save_image(inverse_transform(images), size, image_path)


def vis_square(vis_path, data, type):
    data = (data - data.min()) / (data.max() - data.min())
    n = int(numpy.ceil(numpy.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
                (0, 1), (0, 1)) + ((0, 0),) * (data.ndim - 3))
    data = numpy.pad(data, padding, mode='constant', constant_values=1)
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    plt.imshow(data[:, :, 0])
    plt.axis('off')
    if type:
        plt.savefig('./{}/weights.png'.format(vis_path), format='png')
    else:
        plt.savefie('./{}/activation.png'.format(vis_path), format='png')


class CGAN(object):
    def __init__(self, data_ob, sample_dir, output_size, learn_rate, batch_size, z_dim, y_dim, log_dir, model_path, visua_path):
        self.data_ob = data_ob
        self.sample_dir = sample_dir
        self.output_size = output_size
        self.learn_rate = learn_rate
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.log_dir = log_dir
        self.model_path = model_path
        self.visua_path = visua_path
        self.channels = self.data_ob.shape[2]
        self.images = tensorflow.placeholder(tensorflow.float32, [batch_size, self.output_size, self.output_size, self.channels])
        self.z = tensorflow.placeholder(tensorflow.float32, [self.batch_size, self.z_dim])
        self.y = tensorflow.placeholder(tensorflow.float32, [self.batch_size, self.y_dim])

    def gern_net(self, z, y):
        with tensorflow.variable_scope('generator') as scope:
            yb = tensorflow.reshape(y, shape=[self.batch_size, 1, 1, self.y_dim])
            z = tensorflow.concat([z, y], 1)
            c1, c2 = int(self.output_size / 4), int(self.output_size / 2)
            d1 = tensorflow.nn.relu(batch_normal(fully_connected(z, output_size=1024, scope='gen_fully1'), scope='gen_bn1'))
            d1 = tensorflow.concat([d1, y], 1)
            d2 = tensorflow.nn.relu(batch_normal(fully_connected(d1, output_size=7*7*2*64, scope='gen_fully2'), scope='gen_bn2'))
            d2 = tensorflow.reshape(d2, [self.batch_size, c1, c1, 64 * 2])
            d2 = conv_cond_concat(d2, yb)
            d3 = tensorflow.nn.relu(batch_normal(de_conv2d(d2, output_shape=[self.batch_size, c2, c2, 128], name='gen_deconv1'), scope='gen_bn3'))
            d3 = conv_cond_concat(d3, yb)
            d4 = de_conv2d(d3, output_shape=[self.batch_size, self.output_size, self.output_size, self.channels], name='gen_deconv2', initializer=xavier_initializer())
            return tensorflow.nn.sigmoid(d4)

    def dis_net(self, images, y, reuse=False):
        with tensorflow.variable_scope('discriminator') as scope:
            if reuse == True:
                scope.reuse_variables()

            yb = tensorflow.reshape(y, shape=[self.batch_size, 1, 1, self.y_dim])
            concat_data = conv_cond_concat(images, yb)
            conv1, w1 = conv2d(concat_data, output_dim=10, name='dis_conv1')
            tensorflow.add_to_collection('weight_1', w1)
            conv1 = lrelu(conv1)
            conv1 = conv_cond_concat(conv1, yb)
            tensorflow.add_to_collection('ac_1', conv1)
            conv2, w2 = conv2d(conv1, output_dim=64, name='dis_conv2')
            tensorflow.add_to_collection('weight_2', w2)
            conv2 = lrelu(batch_normal(conv2, scope='dis_bn1'))
            tensorflow.add_to_collection('ac_2', conv2)
            f1 = lrelu(batch_normal(fully_connected(conv2, output_size=1024, scope='dis_fully1'), scope='dis_bn2', reuse=reuse))
            f1 = tensorflow.concat([f1, y], 1)
            out = fully_connected(f1, output_size=1, scope='dis_fully2', initializer=xavier_initializer())
            return tensorflow.nn.sigmoid(out), out

    def test(self):
        init = tensorflow.initialize_all_variables()
        with tensorflow.Session() as sess:
            sess.run(init)
            self.saver.restore(sess, self.model_path)
            sample_z = numpy.random.uniform(1, -1, size=[self.batch_size, self.z_dim])
            output = sess.run(self.fake_images, feed_dict={self.z: sample_z, self.y: sample_label()})
            save_images(output, [8, 8], './{}/test{:02d}_{:04d}.png'.format(self.sample_dir, 0, 0))
            image = cv2.imread('./{}/test{:02d}_{:04d}.png'.format(self.sample_dir, 0, 0), 0)
            cv2.imshow('test', image)
            cv2.waitKey(-1)
            print('Test Finish!')

    def visual(self):
        init = tensorflow.initialize_all_variables()
        with tensorflow.Session() as sess:
            sess.run(init)
            self.saver.restore(sess, self.model_path)
            real_batch_array, real_labels = self.data_ob.getNext_batch(0)
            batch_z = numpy.random.uniform(-1, 1, size=[self.batch_size, self.z_dim])
            conv_weights = sess.run([tensorflow.get_collection('weight_2')])
            vis_square(self.visua_path, conv_weights[0][0].transpose(3, 0, 1, 2), type=1)
            ac = sess.run([tensorflow.get_collection('ac_2')],
                          feed_dict={self.images: real_batch_array[:64], self.z: batch_z, self.y: sample_label()})
            vis_square(self.visua_path, ac[0][0].transpose(3, 1, 2, 0), type=0)
            print('The visualization finish!')

    def build_model(self):
        self.fake_images = self.gern_net(self.z, self.y)
        G_image = tensorflow.summary.image('G_out', self.fake_images)
        D_pro, D_logits = self.dis_net(self.images, self.y, False)
        D_pro_sum = tensorflow.summary.histogram('D_pro', D_pro)
        G_pro, G_logits = self.dis_net(self.fake_images, self.y, True)
        G_pro_sum = tensorflow.summary.histogram('G_pro', G_pro)

        D_fake_loss = tensorflow.reduce_mean(tensorflow.nn.sigmoid_cross_entropy_with_logits(labels=tensorflow.zeros_like(G_pro), logits=G_logits))
        D_real_loss = tensorflow.reduce_mean(tensorflow.nn.sigmoid_cross_entropy_with_logits(labels=tensorflow.ones_like(D_pro), logits=D_logits))
        G_fake_loss = tensorflow.reduce_mean(tensorflow.nn.sigmoid_cross_entropy_with_logits(labels=tensorflow.ones_like(G_pro), logits=G_logits))
        self.D_loss = D_real_loss + D_fake_loss
        self.G_loss = G_fake_loss
        loss_sum = tensorflow.summary.scalar('D_loss', self.D_loss)
        G_loss_sum = tensorflow.summary.scalar('G_loss', self.G_loss)
        self.merged_summary_op_d = tensorflow.summary.merge([loss_sum, D_pro_sum])
        self.merged_summary_op_g = tensorflow.summary.merge([G_loss_sum, G_pro_sum, G_image])
        t_vars = tensorflow.trainable_variables()
        self.d_var = [var for var in t_vars if 'dis' in var.name]
        self.g_var = [var for var in t_vars if 'gen' in var.name]
        self.saver = tensorflow.train.Saver()

    def train(self, epochs=20):
        opti_D = tensorflow.train.AdamOptimizer(learning_rate=self.learn_rate, beta1=5e-1).minimize(self.D_loss, var_list=self.d_var)
        opti_G = tensorflow.train.AdamOptimizer(learning_rate=self.learn_rate, beta1=5e-1).minimize(self.G_loss, var_list=self.g_var)
        init = tensorflow.global_variables_initializer()
        config = tensorflow.ConfigProto()
        config.gpu_options.allow_growth = True
        with tensorflow.Session(config=config) as sess:
            sess.run(init)
            summary_writer = tensorflow.summary.FileWriter(self.log_dir, graph=sess.graph)
            step = 0
            while step < epochs:
                real_batch_array, real_labels = self.data_ob.getNext_batch(step)
                batch_z = numpy.random.uniform(-1, 1, size=[self.batch_size, self.z_dim])
                _, summary_str = sess.run([opti_D, self.merged_summary_op_d],
                                          feed_dict={self.images: real_batch_array, self.z: batch_z, self.y: real_labels})
                summary_writer.add_summary(summary_str, step)
                _, summary_str = sess.run([opti_G, self.merged_summary_op_g],
                                          feed_dict={self.z: batch_z, self.y: real_labels})
                if step % 2 == 0:
                    D_loss = sess.run(self.D_loss, feed_dict={self.images: real_batch_array, self.z: batch_z, self.y: real_labels})
                    fake_loss = sess.run(self.G_loss, feed_dict={self.z: batch_z, self.y: real_labels})
                    print("Step %d: D: loss = %.7f G: loss=%.7f" % (step, D_loss, fake_loss))

                if numpy.mod(step, 50) == 1 and step != 0:
                    sample_images = sess.run(self.fake_images, feed_dict={self.z: batch_z, self.y: sample_label()})
                    save_images(sample_images, [8, 8], './{}/train_{:04d}.png'.format(self.sample_dir, step))
                    self.saver.save(sess, self.model_path)
                step = step + 1
            save_path = self.saver.save(sess, self.model_path)
            print('Model saved in file: %s' % save_path)


