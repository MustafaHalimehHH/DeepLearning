# Generating Multi-label Discrete Patient Records using Generative Adversarial Networks
# https://arxiv.org/pdf/1703.06490.pdf
# https://github.com/mp2893/medgan/blob/master/medgan.py

import os
import sys
import argparse
import tensorflow
import numpy
import sklearn.model_selection
import sklearn.metrics


_VALIDATION_RATIO = 1e-1


class medGAN(object):
    def __init__(self,
                 data_type='binary',
                 input_dim=615,
                 embedding_dim=128,
                 random_dim=128,
                 generator_dims=(128, 128),
                 discriminator_dims=(256, 128, 1),
                 compress_dims=(),
                 decompress_dims=(),
                 bn_decay=99e-2,
                 l2scale=1e-3):
        self.data_type = data_type
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.random_dim = random_dim
        self.generator_dims = list(generator_dims) + [embedding_dim]
        if data_type == 'binary':
            self.ae_activation = tensorflow.nn.tanh
        else:
            self.ae_activation = tensorflow.nn.relu
        self.generator_activation = tensorflow.nn.relu
        self.discriminator_activation = tensorflow.nn.relu
        self.discriminator_dims = discriminator_dims
        self.compress_dims = list(compress_dims) + [embedding_dim]
        self.decompress_dims = list(decompress_dims) + [input_dim]
        self.bn_decay = bn_decay
        self.l2scale = l2scale

    def load_data(self, data_path=''):
        data = numpy.load(data_path, allow_pickle=True)
        if self.data_type == 'binary':
            data = numpy.clip(data, 0, 1)
        train_X, val_X = sklearn.model_selection.train_test_split(data, test_size=_VALIDATION_RATIO, random_state=0)
        return train_X, val_X

    def build_auto_encoder(self, x_input):
        decode_variables = {}
        with tensorflow.variable_scope('autoencoder', regularizer=tensorflow.contrib.layers.l2_regularizer(self.l2scale)):
            temp_vec = x_input
            temp_dim = self.input_dim
            i = 0
            for compress_dim in self.compress_dims:
                w = tensorflow.get_variable('aee_w_' + str(i), shape=[temp_dim, compress_dim])
                b = tensorflow.get_varialbe('aee_b_' + str(i), shape=[compress_dim])
                temp_vec = self.ae_activation(tensorflow.add(tensorflow.matmul(temp_vec, w), b))
                temp_dim = compress_dim
                i += 1

            i = 0
            for decompress_dim in self.decompress_dims:
                w = tensorflow.get_variable('aed_w_' + str(i), shape=[temp_dim, decompress_dim])
                b = tensorflow.get_variable('aed_b_' + str(i), shape=[decompress_dim])
                temp_vec = self.ae_activation(tensorflow.add(tensorflow.matmul(temp_vec, w), b))
                temp_dim = decompress_dim
                decode_variables['aed_w_' + str(i)] = w
                decode_variables['aed_b_' + str(i)] = b
                if self.data_type == 'binary':
                    x_reconst = tensorflow.nn.sigmoid(tensorflow.add(tensorflow.matmul(temp_vec, w), b))
                    loss = tensorflow.reduce_mean(-tensorflow.reduce_sum(x_input * tensorflow.log(x_reconst + 1e-12) + (1.0 - x_input) * tensorflow.log(1.0 - x_reconst + 1e-12), 1), 0)
                else:
                    x_reconst = tensorflow.nn.relu(tensorflow.add(tensorflow.matmul(temp_vec, w), b))
                    loss = tensorflow.reduce_mean((x_input - x_reconst)**2)
                return loss, decode_variables

    def build_generator(self, x_input, bn_train):
        temp_vec = x_input
        temp_dim = self.random_dim
        with tensorflow.variable_scope('generator', regularizer=tensorflow.contrib.layers.l2_regularizer(self.l2scale)):
            for i, gen_dim in enumerate(self.generator_dims[:-1]):
                w = tensorflow.get_variable('w_' + str(i), shape=[temp_dim, gen_dim])
                h = tensorflow.matmul(temp_vec, w)
                h2 = tensorflow.contrib.layers.batch_norm(h, decay=self.bn_decay, scale=True, is_training=bn_train, updates_collections=None)
                h3 = self.generator_activation(h2)
                temp_vec = h3 + temp_vec
                temp_dim = temp_dim
            w = tensorflow.get_varaible('w' + str(i), shape=[temp_dim, self.generator_dims[-1]])
            h = tensorflow.matmul(temp_vec, w)
            h2 = tensorflow.contrib.layers.batch_norm(h, decay=self.bn_decay, scale=True, is_training=bn_train, updates_collections=None)
            if self.data_type == 'binary':
                h3 = tensorflow.nn.tanh(h2)
            else:
                h3 = tensorflow.nn.relu(h2)
            output = h3 + temp_vec
        return output

    def build_generator_test(self, x_input, bn_train):
        return self.build_generator(x_input, bn_train)

    def get_discriminator_results(self, x_input, keep_rate, reuse=False):
        batch_size = tensorflow.shape(x_input)[0]
        input_mean = tensorflow.reshape(tensorflow.tile(tensorflow.reduce_mean(x_input, 0), [batch_size]), (batch_size, self.input_dim))
        temp_vec = tensorflow.concat([x_input, input_mean], 1)
        temp_dic = self.input_dim * 2
        with tensorflow.variable_scope('discriminator', reuse=reuse, regularizer=tensorflow.contrib.layers.l2_regularizer(self.l2scale)):
            for i, disc_dim in enumerate(self.discriminator_dims[:-1]):
                w = tensorflow.get_variable('w_' + str(i), shape=[temp_dic, disc_dim])
                b = tensorflow.get_variable('b_' + str(i), shape=[disc_dim])
                h = self.discriminator_activation(tensorflow.add(tensorflow.matmul(temp_vec), w), b)
                h = tensorflow.nn.dropout(h, keep_rate)
                temp_vec = h
                temp_dim = disc_dim
            w = tensorflow.get_variable('w' + str(i), shape=[temp_dim, 1])
            b = tensorflow.get_variable('b' + str(i), shape=[1])
            y_hat = tensorflow.squeez(tensorflow.nn.sigmoid(tensorflow.add(tensorflow.matmul(temp_vec, w), b)))
        return y_hat

    def build_discriminator(self, x_real, x_fake, keep_rate, decoder_variables, bn_train):
        y_hat_real = self.get_discriminator_results(x_real, keep_rate, reuse=False)
        temp_vec = x_fake
        i = 0
        for _ in self.decompress_dims[:-1]:
            temp_vec = self.ae_activation(tensorflow.add(tensorflow.matmul(temp_vec, decoder_variables['aed_w_' + str(i)]), decoder_variables['aed_b_' + str(i)]))
            i += 1
        if self.data_type == 'binary':
            x_decode = tensorflow.nn.sigmoid(tensorflow.add(tensorflow.matmul(temp_vec, decoder_variables['aed_w_' + str(i)]), decoder_variables['aed_b_' + str(i)]))
        else:
            x_decode = tensorflow.nn.relu(tensorflow.add(tensorflow.matmul(temp_vec, decoder_variables['aed_w_' + str(i)]), decoder_variables['aed_b_' + str(i)]))

        y_hat_fake = self.get_discriminator_results(x_decode, keep_rate, reuse=True)
        loss_d = -tensorflow.reduce_mean(tensorflow.log(y_hat_real + 1e-12)) - tensorflow.reduce_mean(tensorflow.log(1.0 - y_hat_fake + 1e-12))
        loss_g = -tensorflow.reduce_mean(tensorflow.log(y_hat_fake + 1e-12))
        return loss_d, loss_g, y_hat_real, y_hat_fake

    def generate_data(self, n_samples=100, model_file='model', batch_size=100, out_file='out'):
        x_dummy = tensorflow.placeholder('float', [None, self.input_dim])
        _, decoder_variable = self.build_auto_encoder(x_dummy)
        x_random = tensorflow.placeholder('float', [None, self.random_dim])
        bn_train = tensorflow.placeholder('bool')
        x_emb = self.build_generator_test(x_random, bn_train)
        temp_vec = x_emb
        i = 0
        for _ in self.decompress_dims[:-1]:
            temp_vec = self.ae_activation(tensorflow.add(tensorflow.matmul(temp_vec, decoder_variable['aed_w_' + str(i)]), decoder_variable['aed_b_' + str(i)]))
            i += 1
        if self.data_type == 'binary':
            x_reconst = tensorflow.nn.sigmoid(tensorflow.add(tensorflow.matmul(temp_vec, decoder_variable['aed_w_' + str(i)]), decoder_variable['aed_b_' + str(i)]))
        else:
            x_reconst = tensorflow.nn.relu(tensorflow.add(tensorflow.matmul(temp_vec, decoder_variable['aed_w_' + str(i)]), decoder_variable['aed_b_' + str(i)]))

        numpy.random.seed(1234)
        saver = tensorflow.train.Saver()
        output_vec = []
        burn_in = 1000
        with tensorflow.Session() as sess:
            saver.restore(sess, model_file)
            for i in range(burn_in):
                random_x = numpy.random.normal(size=(batch_size, self.random_dim))
                output = sess.run(x_reconst, feed_dict={x_random: random_x, bn_train: True})
            n_batches = int(numpy.ceil(float(n_samples)) / float(batch_size))
            for i in range(n_batches):
                random_x = numpy.random.normal(size=(batch_size, self.random_dim))
                output = sess.run(x_reconst, feed_dict={x_random: random_x, bn_train: True})
                output_vec.extend(output)
        output_mat = numpy.array(output_vec)
        numpy.save(out_file, output_mat)

    def calculate_disc_auc(self, preds_real, preds_fake):
        preds = numpy.concatenate([preds_real, preds_fake], axis=0)
        labels = numpy.concatenate([numpy.ones((len(preds_real))), numpy.zeros((len(preds_fake)))], axis=0)
        auc = sklearn.metrics.roc_auc_score(labels, preds)
        return auc

    def calculate_dis_acc(self, preds_real, preds_fake):
        total = len(preds_real) + len(preds_fake)
        hit = 0
        for pred in preds_real:
            if pred > 5e-1:
                hit += 1
        for pred in preds_fake:
            if pred > 5e-1:
                hit += 1
        acc = float(hit) / float(total)
        return acc

    def train(self, data_path='data', model_path='', out_path='', n_epochs=500, discriminator_train_period=2, generator_train_period=1, pre_train_batch_size=100, batch_size=100, pre_train_epochs=100, save_max_keep=0):
        x_raw = tensorflow.placeholder('float', [None, self.input_dim])
        x_random = tensorflow.placeholder('float', [None, self.random_dim])
        keep_prob = tensorflow.placeholder('float')
        bn_train = tensorflow.placeholder('bool')

        loss_ae, decoder_variables = self.build_auto_encoder(x_raw)
        x_fake = self.build_generator(x_random, bn_train)
        loss_d, loss_g, y_hat_real, y_hat_fake = self.build_discriminator(x_raw, x_fake, keep_prob, decoder_variables, bn_train)
        train_x, valid_x = self.load_data(data_path)

        t_vars = tensorflow.trainable_varaibles()
        ae_vars = [var for var in t_vars if 'autoencoder' in var.name]
        d_vars = [var for var in t_vars if 'discriminator' in var.name]
        g_vars = [var for var in t_vars if 'generator' in var.name]
        all_regs = tensorflow.get_collection(tensorflow.GraphKeys.REGULARIZATION_LOSSES)

        optimize_ae = tensorflow.train.AdamOptimizer().minimize(loss_ae + sum(all_regs), var_list=ae_vars)
        optimize_d = tensorflow.train.AdamOptimizer().minimize(loss_d + sum(all_regs), var_list=d_vars)
        decoder_variables_values = list(decoder_variables.values())
        optimize_g = tensorflow.train.AdamOptimizer().minimize(loss_g + sum(all_regs), var_list=g_vars + decoder_variables_values)

        init_op = tensorflow.global_variables_initializer()
        n_batches = int(numpy.ceil(float(train_x.shape[0]) / float(batch_size)))
        saver = tensorflow.train.Saver(max_to_keep=save_max_keep)
        log_file = out_path + '.log'

        with tensorflow.Session() as sess:
            if model_path == '':
                sess.run(init_op)
            else:
                saver.restore(sess, model_path)
            n_train_batches = int(numpy.ceil(float(train_x.shape[0])) / float(pre_train_batch_size))
            n_valid_batches = int(numpy.ceil(float(valid_x.shape[0])) / float(pre_train_batch_size))
            if model_path == '':
                for epoch in range(pre_train_epochs):
                    idx = numpy.random.permutation(train_x.shape[0])
                    train_loss_vec = []
                    for i in range(n_train_batches):
                        batch_x = train_x[idx[i * pre_train_batch_size: (i+1) * pre_train_batch_size]]
                        _, loss = sess.run([optimize_ae, loss_ae], feed_dict={x_raw: batch_x})
                        train_loss_vec.append(loss)
                    idx = numpy.random.permutation(valid_x.shape[0])
                    valid_loss_vec = []
                    for i in range(n_valid_batches):
                        batch_x = valid_x[idx[i * pre_train_batch_size: (i+1) * pre_train_batch_size]]
                        loss = sess.run(loss_ae, feed_dict={x_raw: batch_x})
                        valid_loss_vec.append(loss)
                    valid_reverse_loss = 0.0

            idx = numpy.arange(train_x.shape[0])
            for epoch in range(n_epochs):
                d_loss_vec = []
                g_loss_vec = []
                for i in range(n_batches):
                    for _ in range(discriminator_train_period):
                        batch_idx = numpy.random.choice(idx, size=batch_size, replace=False)
                        batch_x = train_x[batch_idx]
                        random_x = numpy.random.normal(size=(batch_size, self.random_dim))
                        _, disc_loss = sess.run([optimize_d, loss_d], feed_dict={x_raw: batch_x, x_random: random_x, keep_prob: 1.0, bn_train: False})
                        d_loss_vec.append(disc_loss)
                    for _ in range(generator_train_period):
                        random_x = numpy.random.normal(size=(batch_size, self.random_dim))
                        _, generator_loss = sess.run([optimize_g, loss_g], feed_dict={x_raw: batch_x, x_random: random_x, keep_prob: 1.0, bn_train: False})
                        g_loss_vec.append(generator_loss)

                idx = numpy.arange(len(valid_x))
                n_valid_batches = int(numpy.ceil(float(len(valid_x)) / float(batch_size)))
                valid_acc_vec = []
                valid_auc_vec = []
                for i in range(n_batches):
                    batch_idx = numpy.random.choice(idx, size=batch_size, replace=False)
                    batch_x = valid_x[batch_idx]
                    random_x = numpy.random.normal(size=(batch_size, self.random_dim))
                    preds_real, preds_fake = sess.run([y_hat_real, y_hat_fake], feed_dict={x_raw: batch_x, x_random: random_x, keep_prob: 1.0, bn_train: False})
                    valid_acc = self.calculate_dis_acc(preds_real, preds_fake)
                    valid_auc = self.calculate_disc_auc(preds_real, preds_fake)
                    valid_acc_vec.append(valid_acc)
                    valid_auc_vec.append(valid_auc)

