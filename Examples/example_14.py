from __future__ import print_function, absolute_import, unicode_literals, division
import os
import numpy
import tensorflow
import tensorflow_datasets
import matplotlib.pyplot as plt

print('eager_execution', tensorflow.executing_eagerly())
tensorflow.enable_eager_execution()
print('eager_execution', tensorflow.executing_eagerly())
tensorflow_datasets.disable_progress_bar()

SPLIT_WEIGHTS = (8, 1, 1)
splits = tensorflow_datasets.Split.TRAIN.subsplit(weighted=SPLIT_WEIGHTS)
(raw_train, raw_validation, raw_test), metadata = tensorflow_datasets.load('cats_vs_dogs', split=list(splits), with_info=True, as_supervised=True)
print(raw_train)
print(raw_validation)
print(raw_test)

get_label_image = metadata.features['label'].int2str
print('get_label_image', get_label_image)
for image, label in raw_train.take(2):
    print('loop')
    print('label', get_label_image(label))
    plt.figure()
    plt.imshow(image)
    plt.title(get_label_image(label))
    plt.show()

IMG_SIZE = 160


def format_example(image, label):
    image = tensorflow.cast(image, tensorflow.float32)
    image = (image / 127.5) - 1
    image = tensorflow.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label


train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000
train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)

for image_batch, label_batch in train_batches.take(1):
    pass
print(image_batch.shape, label_batch.shape)

IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
base_model = tensorflow.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
feature_batch = base_model(image_batch)
print('feature_batch', feature_batch.shape)
base_model.trainable = False
global_average_layer = tensorflow.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print('feature_batch_average', feature_batch_average.shape)
prediction_layer = tensorflow.keras.layers.Dense(1)
batch_prediction = prediction_layer(feature_batch_average)
print('batch_prediction', batch_prediction.shape)

model = tensorflow.keras.Sequential([
    base_model,
    global_average_layer,
    prediction_layer
])
model.compile(optimizer=tensorflow.keras.optimizers.RMSprop(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
print(len(model.trainable_variables))

num_train, num_val, num_test = (
    metadata.splits['train'].num_examples * weight/10 for weight in SPLIT_WEIGHTS
)

initial_epochs = 2
steps_per_epoch = round(num_train) // BATCH_SIZE
validation_steps = 20

# loss0, acc0 = model.evaluate(validation_batches, steps=validation_steps)
# print('Initial Loss: {:.2f}'.format(loss0))
# print('Initial Acc: {:.2f}'.format(acc0))

'''
history = model.fit(train_batches,
                    epochs=initial_epochs,
                    validation_data=validation_batches)
acc = history.history['accuracy']
loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']
print('acc', acc)
print('loss', loss)
print('val_acc', val_acc)
print('val_loss', val_loss)
'''

base_model.trainable = True
print('Number of layers in the base model: ', len(base_model.layers))
fine_tune_at = 100
# Freezing all the layers before the fine tune layer
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False
    print('layer', layer.name)

model.compile(optimizer=tensorflow.keras.optimizers.RMSprop(lr=1e-4), metrics=['accuracy'], loss='binary_crossentropy')
print(len(model.trainable_variables))
