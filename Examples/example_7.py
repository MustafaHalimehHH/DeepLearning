# https://www.kaggle.com/ibtesama/pneumonia-detection-fine-tuning-and-cam
import os
import glob
import pickle
from pathlib import Path
import gc
import numpy
import imgaug
import pandas
import seaborn
import keras
import tensorflow
import sklearn
import cv2
import skimage
import matplotlib.pyplot as plt

color = seaborn.color_palette()
os.environ['PYTHONHASHSEED'] = '0'
numpy.random.seed(111)
session_conf = tensorflow.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
tensorflow.set_random_seed(111)
sess = tensorflow.Session(graph=tensorflow.get_default_graph(), config=session_conf)
keras.backend.set_session(sess)
imgaug.seed(111)

data_dir = Path('D:\Halimeh\Datasets\Kaggle\chest-xray-pneumonia\chest_xray')
train_dir = data_dir / 'train'
val_dir = data_dir / 'val'
test_dir = data_dir / 'test'


def load_train():
    normal_cases_dir = train_dir / 'NORMAL'
    pneumonia_cases_dir = train_dir / 'PENUMONIA'
    normal_cases = normal_cases_dir.glob('*.jpeg')
    pneumonia_cases = pneumonia_cases_dir.glob('*.jpeg')
    train_data = []
    train_label = []
    for img in normal_cases:
        train_data.append(img)
        train_label.append(0)
    for img in pneumonia_cases:
        train_data.append(img)
        train_label.append(1)
    df = pandas.DataFrame(train_data)
    df.columns = ['images']
    df['labels'] = train_label
    df = df.sample(frac=1).reset_index(drop=True)
    return df


def prepare_val_test(is_val=True):
    if is_val:
        normal_dir = val_dir / 'NORMAL'
        pneumonia_dir = val_dir / 'PNEUMONIA'
    else:
        normal_dir = test_dir / 'NORMAL'
        pneumonia_dir = test_dir / 'PNEUMONIA'
    normal_cases = normal_dir.glob("*.jpeg")
    pneumonia_cases = pneumonia_dir.glob("*.jpeg")
    data, labels = ([] for x in range(2))

    def prepare_img(imgs_case):
        for img in imgs_case:
            img = cv2.imread(str(img))
            img = cv2.resize(img, (224, 224))
            if img.shape[2] == 1:
                img = numpy.dstack([img, img, img])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(numpy.float32) / 255.0
            if imgs_case == normal_cases:
                label = keras.utils.to_categorical(0, num_classes=2)
            else:
                label = keras.utils.to_categorical(1, num_classes=2)
            data.append(img)
            labels.append(label)
            return data, labels

    prepare_img(normal_cases)
    data, labels = prepare_img(pneumonia_cases)
    data_arr = numpy.array(data)
    labels_arr = numpy.array(labels)
    return data_arr, labels_arr


def data_gen(data, batch_size):
    seq = imgaug.augmenters.OneOf([
        imgaug.augmenters.Fliplr(),
        imgaug.augmenters.Affine(rotate=20),
        imgaug.augmenters.Multiply((1.2, 1.5))
    ])

    n = len(data)
    n_steps = n // batch_size
    batch_data = numpy.zeros((batch_size, 224, 224, 3), dtype=numpy.float32)
    batch_label = numpy.zeros((batch_size, 2), dtype=numpy.float32)
    indices = numpy.arange(n)
    i = 0
    while True:
        numpy.random.shuffle(indices)
        count = 0
        next_batch = indices[(i) * batch_size:(i+1) * batch_size]
        for j, idx in enumerate(next_batch):
            img_name = data.iloc[idx]['images']
            label = data.iloc[idx]['labels']
            encoded_label = keras.utils.to_categorical(label, num_classes=2)
            img = cv2.imread(str(img_name))
            img = cv2.resize(img, (224, 224))
            if img.shape[2] == 1:
                img = numpy.dstack([img, img, img])
            orig_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            orig_img = orig_img.astype(numpy.float32) / 255.0
            batch_data[count] = orig_img
            batch_label[count] = encoded_label

            if label == 0 and count < batch_size - 2:
                aug_img1 = seq.augment_image(img)
                aug_img2 = seq.augment_image(img)
                aug_img1 = cv2.cvtColor(aug_img1, cv2.COLOR_BGR2RGB)
                aug_img2 = cv2.cvtColor(aug_img2, cv2.COLOR_BGR2RGB)
                aug_img1 = aug_img1.astype(numpy.float32) / 255.0
                aug_img2 = aug_img2.astype(numpy.float32) / 255.0
                batch_data[count+1] = aug_img1
                batch_label[count+1] = encoded_label
                batch_data[count+2] = aug_img2
                batch_label[count+2] = encoded_label
                count += 3
            else:
                count += 1
            if count == batch_size - 1:
                break
        i += 1
        yield batch_data, batch_label
        if i > n_steps:
            i = 0


def vgg16_model():
    model = keras.applications.VGG16(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
    x = keras.layers.Dense(1024, activation='relu')(model.layers[-4].output)
    x = keras.layers.Dropout(0.7)(x)
    x = keras.layers.Dense(512, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(2, activation='sigmoid')(x)
    model = keras.models.Model(model.input, x)
    model.summary()
    return model


train_data = load_train()
print('train_data', len(train_data))
val_data, val_labels = prepare_val_test(is_val=True)
test_data, test_labels = prepare_val_test(is_val=False)


vgg_conv = vgg16_model()
for layer in vgg_conv.layers[:-10]:
    layer.trainable = False
for layer in vgg_conv.layers:
    print(layer, layer.trainable)

'''
opt = keras.optimizers.Adam(lr=1e-4, decay=1e-5)
early_stop = keras.callbacks.EarlyStopping(monitor='loss', patience=3, verbose=1)
check_point = keras.callbacks.ModelCheckpoint('kaggle_1', save_best_only=True, save_weights_only=True)
vgg_conv.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=opt)

batch_size = 16
nb_epochs = 2

train_data_gen = data_gen(data=train_data, batch_size=batch_size)
nb_train_steps = train_data.shape[0] // batch_size
print("Number of training and validation steps: {} and {}".format(nb_train_steps, len(val_data)))
history = vgg_conv.fit_generator(
    train_data_gen,
    epochs=nb_epochs,
    steps_per_epoch=nb_train_steps,
    validation_data=(val_data, val_labels),
    callbacks=[early_stop, check_point],
    class_weight={0:1.0, 1:0.4}
)
with open('history_kaggle_1', 'wb') as history_file:
    pickle.dump(history.history, history_file)
'''
vgg_conv.load_weights('kaggle_1')
print('weights loaded')
print('test_data', test_data.shape)
import innvestigate
from innvestigate.utils.visualizations import heatmap

# model_wos = innvestigate.utils.model_wo_softmax(vgg_conv)lrp.sequential_preset_a_flat
analyzer = innvestigate.create_analyzer('lrp.alpha_1_beta_0', vgg_conv)
# analyzer = innvestigate.create_analyzer('deep_taylor', vgg_conv)
# analyzer = innvestigate.create_analyzer('lrp.z_plus', vgg_conv)
# analyzer = innvestigate.create_analyzer('lrp.epsilon', vgg_conv)
sample = test_data[0]
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