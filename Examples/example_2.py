import os
import numpy
import keras
import pylab as plt


def generate_movies(n_samples=200, n_frames=15):
    row = 80
    col = 80
    noisy_movies = numpy.zeros((n_samples, n_frames, row, col, 1), dtype=numpy.float)
    shifted_movies = numpy.zeros((n_samples, n_frames, row, col, 1), dtype=numpy.float)

    for i in range(n_samples):
        num_squares = numpy.random.randint(3, 8)
        for j in range(num_squares):
            # Initial Position
            x_start = numpy.random.randint(20, 60)
            y_start = numpy.random.randint(20, 60)

            # Direction of motion
            x_direction = numpy.random.randint(0, 3) - 1
            y_direction = numpy.random.randint(0, 3) - 1

            # Square size
            s = numpy.random.randint(2, 4)
            for f in range(n_frames):
                x_shift = x_start + (x_direction * f)
                y_shift = y_start + (y_direction * f)
                noisy_movies[i, f, x_shift - s:x_shift + s, y_shift - s:y_shift+s, 0] += 1

                if numpy.random.randint(0, 2):
                    noise_f = (-1)**numpy.random.randint(0,2)
                    noisy_movies[i, f, x_shift - s - 1:x_shift + s + 1, y_shift - s - 1:y_shift + s + 1, 0] += noise_f * 0.1

                x_shift = x_start + x_direction * (f + 1)
                y_shift = y_start + y_direction * (f + 1)
                shifted_movies[i, f, x_shift - s:x_shift + s, y_shift - s:y_shift + s, 0] += 1

    noisy_movies = noisy_movies[::, ::, 20:60, 20:60, ::]
    shifted_movies = shifted_movies[::, ::, 20:60, 20:60, ::]
    noisy_movies[noisy_movies > 1] = 1
    shifted_movies[shifted_movies > 1] = 1
    return noisy_movies, shifted_movies


noisy_movies, shifted_movies = generate_movies(400)
print('noisy_movies', noisy_movies.shape)
print('shifted_movies', shifted_movies.shape)

rnn_model = keras.models.Sequential()
rnn_model.add(keras.layers.ConvLSTM2D(filters=40, kernel_size=(3, 3), input_shape=(None, 40, 40, 1), padding='same', return_sequences=True))
rnn_model.add(keras.layers.BatchNormalization())
rnn_model.add(keras.layers.ConvLSTM2D(filters=40, kernel_size=(3, 3), padding='same', return_sequences=True))
rnn_model.add(keras.layers.BatchNormalization())
rnn_model.add(keras.layers.ConvLSTM2D(filters=40, kernel_size=(3, 3), padding='same', return_sequences=True))
rnn_model.add(keras.layers.BatchNormalization())
rnn_model.add(keras.layers.ConvLSTM2D(filters=40, kernel_size=(3, 3), padding='same', return_sequences=True))
rnn_model.add(keras.layers.BatchNormalization())
rnn_model.add(keras.layers.Conv3D(filters=1, kernel_size=(3, 3, 3), activation='sigmoid', padding='same', data_format='channels_last'))

rnn_model.compile(optimizer='adadelta', loss='binary_crossentropy')

history = rnn_model.fit(noisy_movies, shifted_movies, batch_size=10, epochs=2, validation_split=0.05)
