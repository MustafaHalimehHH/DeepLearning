import os
import glob
import pickle
import h5py
import shutil
import numpy
import pandas
import seaborn
from PIL import Image
import skimage
import keras
import sklearn
import sklearn.metrics
from sklearn.metrics import confusion_matrix
import tensorflow
import cv2
import imgaug
from pathlib import Path
import matplotlib.pylab as plt

os.environ['PYTHONHASHSEED'] = '0'
numpy.random.seed(111)
tensorflow.set_random_seed(111)
session_conf = tensorflow.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tensorflow.Session(graph=tensorflow.get_default_graph(), config=session_conf)
keras.backend.set_session(sess)

data_dir = Path('D:\Halimeh\Datasets\Kaggle\chest-xray-pneumonia\chest_xray')
train_dir = data_dir / 'train'
val_dir = data_dir / 'val'
test_dir = data_dir / 'test'

normal_dir = train_dir / 'NORMAL'
pneumonia_dir = train_dir / 'PNEUMONIA'

normal_cases = normal_dir.glob('*.jpeg')
pneumonia_cases = pneumonia_dir.glob('*.jpeg')

train_data = []
for img in normal_cases:
    train_data.append((img, 0))
for img in pneumonia_cases:
    train_data.append((img, 1))

train_data = pandas.DataFrame(train_data, columns=['image', 'label'], index=None)
train_data = train_data.sample(frac=1.).reset_index(drop=True)

cases_count = train_data['label'].value_counts()
print('cases_count', cases_count)
plt.figure(figsize=(10, 8))
seaborn.barplot(x=cases_count.index, y=cases_count.values)
plt.title('Number of cases')
plt.xlabel('Case type')
plt.ylabel('Count')
plt.xticks(range(len(cases_count.index)), ['Normal(0)', 'Pneumonia(1)'])
plt.show()

normal_samples = (train_data[train_data['label'] == 0]['image'].iloc[:5]).tolist()
pneumonia_samples = (train_data[train_data['label'] == 1]['image'].iloc[:5]).tolist()
samples = normal_samples + pneumonia_samples
fig, ax = plt.subplots(2, 5, figsize=(24, 8))
for i in range(10):
    img = skimage.io.imread(samples[i])
    ax[i // 5, i % 5].imshow(img, cmap='gray')
    ax[i // 5, i % 5].axis('off')
    ax[i // 5, i % 5].set_aspect('auto')
    if i < 5:
        ax[i // 5, i % 5].set_title('Pneumonia')
    else:
        ax[i // 5, i % 5].set_title('Normal')
plt.show()

val_normal_dir = val_dir / 'NORMAL'
val_pneumonia_dir = val_dir / 'PNEUMONIA'
val_normal_cases = val_normal_dir.glob('*.jpeg')
val_pneumonia_cases = val_pneumonia_dir.glob('*.jpeg')
val_data = []
val_labels = []
for img in val_normal_cases:
    img = cv2.imread(str(img))
    img = cv2.resize(img, (224, 224))
    if img.shape[2] == 1:
        img = numpy.dstack([img, img, img])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(numpy.float32) / 255.0
    label = keras.utils.to_categorical(0, num_classes=2)
    val_data.append(img)
    val_labels.append(label)
for img in val_pneumonia_cases:
    img = cv2.imread(str(img))
    img = cv2.resize(img, (224, 224))
    if img.shape[2] == 1:
        img = numpy.dstack([img, img, img])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(numpy.float32) / 255.0
    label = keras.utils.to_categorical(1, num_classes=2)
    val_data.append(img)
    val_labels.append(label)
val_data = numpy.array(val_data)
val_labels = numpy.array(val_labels)
print('val_data', val_data.shape)
print('val_labels', val_labels.shape)


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = sklearn.metrics.confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[sklearn.utils.multiclass.unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=numpy.arange(cm.shape[1]),
           yticks=numpy.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def data_gen(data, batch_size):
    seq = imgaug.augmenters.OneOf([
        imgaug.augmenters.Fliplr(),
        imgaug.augmenters.Affine(rotate=20),
        imgaug.augmenters.Multiply((1.2, 1.5))
    ])

    n = len(data)
    steps = n // batch_size
    batch_data = numpy.zeros((batch_size, 224, 224, 3), dtype=numpy.float32)
    batch_labels = numpy.zeros((batch_size, 2), dtype=numpy.float32)
    indices = numpy.arange(n)
    i = 0
    while True:
        numpy.random.shuffle(indices)
        count = 0
        next_batch = indices[(i * batch_size):(i + 1) * batch_size]
        for j, idx in enumerate(next_batch):
            img_name = data.iloc[idx]['image']
            label = data.iloc[idx]['label']
            enc_label = keras.utils.to_categorical(label, num_classes=2)
            img = cv2.imread(str(img_name))
            img = cv2.resize(img, (224, 224))
            if img.shape[2] == 1:
                img = numpy.dstack([img, img, img])
            b_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            b_img = img.astype(numpy.float32) / 255.0
            batch_data[count] = b_img
            batch_labels[count] = enc_label

            if label == 0 and count < batch_size - 2:
                aug_img_1 = seq.augment_image(img)
                aug_img_2 = seq.augment_image(img)
                aug_img_1 = cv2.cvtColor(aug_img_1, cv2.COLOR_BGR2RGB)
                aug_img_2 = cv2.cvtColor(aug_img_2, cv2.COLOR_BGR2RGB)
                aug_img_1 = aug_img_1.astype(numpy.float32) / 255.0
                aug_img_2 = aug_img_2.astype(numpy.float32) / 255.0
                batch_data[count + 1] = aug_img_1
                batch_labels[count + 1] = enc_label
                batch_data[count + 2] = aug_img_2
                batch_labels[count + 2] = enc_label
                count += 2
            else:
                count += 1
            if count == batch_size - 1:
                break

        i += 1
        yield batch_data, batch_labels
        if i >= steps:
            i = 0


def build_model():
    input_img = keras.layers.Input(shape=(224, 224, 3), name='ImageInput')
    x = keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu', name='Conv1_1')(input_img)
    x = keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu', name='Conv1_2')(x)
    x = keras.layers.MaxPool2D((2, 2), name='pool1')(x)

    x = keras.layers.SeparableConv2D(128, kernel_size=(3, 3), activation='relu', padding='same', name='Conv2_1')(x)
    x = keras.layers.SeparableConv2D(128, kernel_size=(3, 3), activation='relu', padding='same', name='Conv2_2')(x)
    x = keras.layers.MaxPooling2D((2, 2), name='pool2')(x)

    x = keras.layers.SeparableConv2D(256, kernel_size=(3, 3), activation='relu', padding='same', name='Conv3_1')(x)
    x = keras.layers.BatchNormalization(name='bn1')(x)
    x = keras.layers.SeparableConv2D(256, (3, 3), activation='relu', padding='same', name='Conv3_2')(x)
    x = keras.layers.BatchNormalization(name='bn2')(x)
    x = keras.layers.SeparableConv2D(256, kernel_size=(3, 3), activation='relu', padding='same', name='Conv3_3')(x)
    x = keras.layers.MaxPooling2D((2, 2), name='pool3')(x)

    x = keras.layers.SeparableConv2D(512, kernel_size=(3, 3), activation='relu', padding='same', name='Conv4_1')(x)
    x = keras.layers.BatchNormalization(name='bn3')(x)
    x = keras.layers.SeparableConv2D(512, kernel_size=(3, 3), activation='relu', padding='same', name='Conv4_2')(x)
    x = keras.layers.BatchNormalization(name='bn4')(x)
    x = keras.layers.SeparableConv2D(512, kernel_size=(3, 3), activation='relu', padding='same', name='Conv4_3')(x)
    x = keras.layers.MaxPooling2D((2, 2), name='pool4')(x)

    x = keras.layers.Flatten(name='flatten')(x)
    x = keras.layers.Dense(1024, activation='relu', name='fc1')(x)
    x = keras.layers.Dropout(0.7, name='dropout1')(x)
    x = keras.layers.Dense(512, activation='relu', name='fc2')(x)
    x = keras.layers.Dropout(0.5, name='dropout2')(x)
    x = keras.layers.Dense(2, activation='softmax', name='fc3')(x)

    model = keras.models.Model(inputs=input_img, outputs=x)
    model.summary()
    return model


def initialize_weights(weights_path, model):
    f = h5py.File(weights_path, 'r')

    w, b = f['block1_conv1']['block1_conv1_W_1:0'], f['block1_conv1']['block1_conv1_b_1:0']
    model.layers[1].set_weights = [w, b]
    w, b = f['block1_conv2']['block1_conv2_W_1:0'], f['block1_conv2']['block1_conv2_b_1:0']
    model.layers[2].set_weights = [w, b]
    w, b = f['block2_conv1']['block2_conv1_W_1:0'], f['block2_conv1']['block2_conv1_b_1:0']
    model.layers[4].set_weights = [w, b]
    w, b = f['block2_conv2']['block2_conv2_W_1:0'], f['block2_conv2']['block2_conv2_b_1:0']
    model.layers[5].set_weights = [w, b]

    f.close()


model = build_model()
initialize_weights('D:\\Halimeh\\Datasets\\Kaggle\\Pre-Traind\\vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                   model)
es = keras.callbacks.EarlyStopping(patience=5)
chkpt = keras.callbacks.ModelCheckpoint(filepath='best_model_todate', save_best_only=True, save_weights_only=True)
model.compile(optimizer=keras.optimizers.Adam(lr=1e-4, decay=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
batch_size = 16
n_epochs = 1
train_data_gen = data_gen(data=train_data, batch_size=batch_size)
n_train_steps = train_data.shape[0] // batch_size
print('Number of training and validation steps: {} and {} '.format(n_train_steps, len(val_data)))
'''
history = model.fit_generator(
    train_data_gen,
    class_weight={0:1.0, 1:0.4},
    epochs=n_epochs,
    steps_per_epoch=n_train_steps,
    validation_data=[val_data, val_labels],
    callbacks=[chkpt, es]
)
with open('history_kaggle', 'wb') as history_file:
    pickle.dump(history.history, history_file)
'''

model.load_weights('best_model_todate')
print('weights loaded')

normal_dir = test_dir / 'NORMAL'
pneumonia_dir = test_dir / 'PNEUMONIA'
normal_cases = normal_dir.glob('*.jpeg')
pneumonia_cases = pneumonia_dir.glob('*.jpeg')
test_data = []
test_label = []
for img in normal_cases:
    img = cv2.imread(str(img))
    img = cv2.resize(img, (224, 224))
    if img.shape[2] == 1:
        img = numpy.dstack([img, img, img])
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(numpy.float32) / 255.0
    label = keras.utils.to_categorical(0, num_classes=2)
    test_data.append(img)
    test_label.append(label)
for img in pneumonia_cases:
    img = cv2.imread(str(img))
    img = cv2.resize(img, (224, 224))
    if img.shape[2] == 1:
        img = numpy.dstack([img, img, img])
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(numpy.float32) / 255.0
    label = keras.utils.to_categorical(1, num_classes=2)
    test_data.append(img)
    test_label.append(label)

test_data = numpy.array(test_data)
test_label = numpy.array(test_label)
print('Test data', test_data.shape, test_label.shape)
# test_loss, test_score = model.evaluate(test_data, test_label, batch_size=16)
# print('TEST Loss', test_loss)
# print('TEST Accuracy', test_score)

import innvestigate
from innvestigate.utils.visualizations import heatmap
model_wos = innvestigate.utils.model_wo_softmax(model)
# analyzer = innvestigate.create_analyzer('lrp.z_plus', model_wos)
analyzer = innvestigate.create_analyzer('lrp.epsilon', model_wos)
sample = test_data[1]
print('sample', sample.shape)
sample = numpy.expand_dims(sample, axis=0)
a = analyzer.analyze(sample)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 6))
ax1.imshow(test_data[0])
ax1.axis('off')
ax2.imshow(heatmap(a)[0], interpolation='none')
ax2.axis('off')
plt.show()
exit(1)
preds = model.predict(test_data, batch_size=16)
preds = numpy.argmax(preds, axis=-1)
orig_test_labels = numpy.argmax(test_label, axis=-1)
print('preds', preds.shape, preds)
print('orig_test_labels', orig_test_labels.shape, orig_test_labels)
print('preds bincount', numpy.bincount(preds))
print('trues bincount', numpy.bincount(orig_test_labels))
# exit(1)

# cm = sklearn.metrics.confusion_matrix(orig_test_labels, preds)
# plot_confusion_matrix(orig_test_labels, preds, classes=['NORMAL', 'PNEUMONIA'], title='cm', normalize=True)
# plt.show()
