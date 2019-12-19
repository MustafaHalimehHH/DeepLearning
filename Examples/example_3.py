from __future__ import absolute_import, division, print_function, unicode_literals
import os
import tempfile
import numpy
import pandas
import tensorflow
from tensorflow import keras
import sklearn
import sklearn.model_selection
import seaborn
import matplotlib
import matplotlib.pylab as plt

matplotlib.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

raw_df = pandas.read_csv('https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv')
print(raw_df.head())

neg, pos = numpy.bincount(raw_df['Class'])
total = neg + pos
print('Examples:\n  Total: {}\n     Positive: {} ({:.2f}% of total)\n'.format(total, pos, 100 * pos/total))

raw_df.pop('Time')
raw_df['Log Amount'] = numpy.log(raw_df.pop('Amount') + 1e-3)

train_df, test_df = sklearn.model_selection.train_test_split(raw_df, test_size=0.2)
train_df, val_df = sklearn.model_selection.train_test_split(train_df, test_size=0.2)

train_labels = numpy.array(train_df.pop('Class'))
bool_train_labels = train_labels != 0
val_labels = numpy.array(val_df.pop('Class'))
test_labels = numpy.array(test_df.pop('Class'))

train_features = numpy.array(train_df)
val_features = numpy.array(val_df)
test_features = numpy.array(test_df)

standard_scaler = sklearn.preprocessing.StandardScaler()
train_features = standard_scaler.fit_transform(train_features)
val_features = standard_scaler.transform(val_features)
test_features = standard_scaler.transform(test_features)

train_features = numpy.clip(train_features, -5, 5)
val_features = numpy.clip(val_features, -5, 5)
test_features = numpy.clip(test_features, -5, 5)

METRICS = [
    keras.metrics.TruePositives(name='tp'),
    keras.metrics.FalsePositives(name='fp'),
    keras.metrics.TrueNegatives(name='tn'),
    keras.metrics.FalseNegatives(name='fn'),
    keras.metrics.BinaryAccuracy(name='accuracy'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
    keras.metrics.AUC(name='auc')
]


def make_model(metrics=METRICS, output_bias=None):
    if output_bias is not None:
        output_bias = tensorflow.keras.initializers.Constant(output_bias)
    model = keras.Sequential([
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias)
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(lr=1e-3),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=METRICS
    )

    return model


def plot_loss(history, label, n):
    plt.semilogy(history.epoch, history.history['loss'], color=colors[n], label='Train '+label)
    plt.semilogy(history.epoch, history.history['val_loss'], color=colors[n], label='Val '+label, linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def plot_metrics(history):
    metrics = ['loss', 'auc', 'precision', 'recall']
    for n, metric in enumerate(metrics):
        name = metric.replace("_", " ").capitalize()
        plt.subplot(2, 2, n+1)
        plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
        plt.plot(history.epoch, history.history['val_'+metric], color=colors[0], label='Val', linestyle='--')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        plt.legend()
    plt.show()


def plot_roc(name, labels, predictions, **k):
    fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)
    plt.plot(100*fp, 100*tp, label=name, linewidth=2, **k)
    plt.xlabel('False Positives [%]')
    plt.ylabel('True Positives [%]')
    plt.grid(True)
    plt.legend()
    plt.gca().set_aspect('equal')
    plt.show()


EPOCHS = 10
BATCH_SIZE = 2048

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_auc',
    verbose=1,
    patience=10,
    mode='max',
    restore_best_weights=True
)

model = make_model()
history = model.fit(
    train_features,
    train_labels,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(val_features, val_labels)
)

train_predictions = model.predict(train_features, batch_size=BATCH_SIZE)

plot_loss(history, 'Model', 0)
plot_metrics(history)
plot_roc("Train", train_labels, train_predictions, color=colors[0])