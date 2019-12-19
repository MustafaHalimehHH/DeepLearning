# https://github.com/wiseodd/generative-models/blob/master/GAN/wasserstein_gan/wgan_tensorflow.py
import os
import numpy
import tensorflow
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

mb_size = 32
x_dim = 784
z_dim = 10
h_dim = 128

mnist = tensorflow.keras.datasets.mnist.load_data()


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=5e-2, hspace=5e-2)
    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
    return fig


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1.0 / tensorflow.sqrt(in_dim / 2.0)
    return tensorflow.random_normal(shape=size, stddev=xavier_stddev)


X = tensorflow.placeholder(tensorflow.float32, shape=[None, x_dim])
D_W1 = tensorflow.variable(xavier_init([x_dim, h_dim]))
D_B1 = tensorflow.variable(tensorflow.zeros(shape=[h_dim]))
D_W2 = tensorflow.variable(xavier_init([h_dim, 1]))
D_B2 = tensorflow.variable(tensorflow.zeros(shape=[1]))
theta_D = [D_W1, D_W2, D_B1, D_B2]

z = tensorflow.placeholder(tensorflow.float32, shape=[None, z_dim])
G_W1 = tensorflow.variable(xavier_init([z_dim, h_dim]))
G_B1 = tensorflow.variable(tensorflow.zeros(shape=[h_dim]))
G_W2 = tensorflow.variable(xavier_init([h_dim, x_dim]))
G_B2 = tensorflow.variable(tensorflow.zeros(shape=[x_dim]))
theta_G = [G_W1, G_W2, G_B1, G_B2]


def sample_z(m, n):
    return numpy.random.uniform(-1, 1, size=[m, n])


def generator(z):
    G_h1 = tensorflow.nn.relu(tensorflow.matmul(z, G_W1) + G_B1)
    G_log_prob = tensorflow.matmul(G_h1, G_W2) + G_B2
    G_prob = tensorflow.nn.sigmoid(G_log_prob)
    return G_prob


def discriminator(x):
    D_h1 = tensorflow.nn.relu(tensorflow.matmul(x, D_W1) + D_B1)
    out = tensorflow.matmul(D_h1, D_W2) + D_B2
    return out


G_sample = generator(z)
D_real = discriminator(X)
D_fake = discriminator(G_sample)
D_loss = tensorflow.reduce_mean(D_real) - tensorflow.reduce_mean(D_fake)
G_loss = -tensorflow.reduce_mean(D_fake)

D_solver = (tensorflow.train.RMSPropOptimizer(learning_rate=1e-4).minimize(-D_loss, var_list=theta_D))
G_solver = (tensorflow.train.RMSPropOptimizer(learning_rate=1e-4).minimize(G_loss, var_list=theta_G))
clip_D = [p.assign(tensorflow.clip_by_value(p, -0.01, 0.01)) for p in theta_D]

sess = tensorflow.Session()
sess.run(tensorflow.global_variables_initializer())

i = 0
for i in range(100):
    for _ in range(5):
        X_mb, _ = mnist.train.next_batch(mb_size)
        _, D_loss_curr, _ = sess.run([D_solver, D_loss, clip_D],
                                     feed_dict={X: X_mb, z: sample_z(mb_size, z_dim)})
    _, G_loss_curr = sess.run([G_solver, G_loss],
                              feed_dict={z: sample_z(mb_size, z_dim)})
