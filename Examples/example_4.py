import os
import numpy
import pandas
import glob
import sklearn
import tensorflow
import keras
import sklearn.model_selection
import nibabel
import skimage
import skimage.util
import warnings
import matplotlib.pylab as plt


BASE_IMG_PATH = 'D:\Halimeh\Datasets\Kaggle'
DS_FACT = 8
print('keras', keras.__version__)
print('tensorflow', tensorflow.__version__)


def montage_nd(in_img):
    if len(in_img.shape) > 3:
        return skimage.util.montage(numpy.stack([montage_nd(x_slice) for x_slice in in_img], 0))
    elif len(in_img.shape) == 3:
        return skimage.util.montage(in_img)
    else:
        warnings.warn('Input less than 3d image, no processing')
        return in_img


def read_all_slices(in_paths, rescale=True):
    cur_vol = numpy.expand_dims(numpy.concatenate([nibabel.load(c_path).get_data()[:, ::DS_FACT, ::DS_FACT] for c_path in in_paths], 0), -1)
    if rescale:
        return (cur_vol.astype(numpy.float32) + 500) / 2000.0
    else:
        return cur_vol/255.0


def read_both(in_paths):
    in_vol = read_all_slices(in_paths)
    in_mask = read_all_slices(map(lambda x: x.replace('IMG_', 'MASK_'), in_paths), rescale=False)
    return in_vol, in_mask


def gen_chunk(in_img, in_mask, slice_count=10, batch_size=16):
    while True:
        img_batch = []
        mask_batch = []
        for _ in range(batch_size):
            inx = numpy.random.choice(range(in_img.shape[0] - slice_count))
            img_batch += [in_img[inx:(inx + slice_count)]]
            mask_batch += [in_mask[inx:(inx + slice_count)]]
        yield numpy.stack(img_batch, 0), numpy.stack(mask_batch, 0)


def gen_aug_chunk(in_gen):
    for i, (x_img, y_img) in enumerate(in_gen):
        xy_block = numpy.concatenate([x_img, y_img], 1).swapaxes(1, 4)[:, 0]
        img_gen = d_gen.flow(xy_block, shuffle=True, seed=i, batch_size=xy_block.shape[0])
        xy_scat = next(img_gen)
        xy_scat = numpy.expand_dims(xy_scat, 1).swapaxes(1 ,4)
        yield xy_scat[:, :xy_scat.shape[1]//2], xy_scat[:, xy_scat.shape[1]//2:]


def make_model():
    sim_model = tensorflow.keras.models.Sequential()
    sim_model.add(tensorflow.keras.layers.BatchNormalization(input_shape=(None, None, None, 1)))
    sim_model.add(tensorflow.keras.layers.Conv3D(8, kernel_size=(1, 5, 5), padding='same', activation='relu'))
    sim_model.add(tensorflow.keras.layers.Conv3D(8, kernel_size=(3, 3, 3), padding='same', activation='relu'))
    sim_model.add(tensorflow.keras.layers.BatchNormalization())
    sim_model.add(tensorflow.keras.layers.Bidirectional(layer=tensorflow.keras.layers.ConvLSTM2D(16, kernel_size=(3, 3), padding='same', return_sequences=True)))
    sim_model.add(tensorflow.keras.layers.Bidirectional(layer=tensorflow.keras.layers.ConvLSTM2D(32, kernel_size=(3, 3), padding='same', return_sequences=True)))
    sim_model.add(tensorflow.keras.layers.Conv3D(8, kernel_size=(1, 3, 3), padding='same', activation='relu'))
    sim_model.add(tensorflow.keras.layers.Conv3D(1, kernel_size=(1, 1, 1), activation='sigmoid'))
    '''
    sim_model.add(tensorflow.keras.layers.Cropping3D((1, 2, 2)))
    sim_model.add(tensorflow.keras.layers.ZeroPadding3D((1, 2, 2)))
    '''
    sim_model.summary()
    return sim_model


all_images = glob.glob(os.path.join(BASE_IMG_PATH, '3d_images', 'IMG_*'))
print(len(all_images), ' matching files found:', all_images[0])
train_paths, test_paths = sklearn.model_selection.train_test_split(all_images, random_state=2019, test_size=0.5)
print(len(train_paths), 'training size')
print(len(test_paths), 'test size')


train_vol, train_mask = read_both(train_paths)
test_vol, test_mask = read_both(test_paths)
print('train', train_vol.shape, 'mask', train_mask.shape)
print('test', test_vol.shape, 'mask', test_mask.shape)
plt.hist(train_vol.ravel(), numpy.linspace(-1, 1, 50))
plt.show()


train_gen = gen_chunk(train_vol, train_mask)
valid_gen = gen_chunk(test_vol, test_mask, slice_count=100, batch_size=1)
'''
x_out, y_out = next(train_gen)
print(x_out.shape, y_out.shape)
# x_out[0, :, ...] = 0
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
ax1.imshow(montage_nd(x_out[..., 0]), cmap='bone')
ax1.set_title('In Batch')
ax2.imshow(montage_nd(y_out[..., 0]))
ax2.set_title('Out Batch')
plt.show()
'''

d_gen = tensorflow.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.1,
    zoom_range=0.25,
    fill_mode='nearest',
    horizontal_flip=True,
    vertical_flip=False
)


train_aug_gen = gen_aug_chunk(train_gen)

# x_out, y_out = next(train_aug_gen)
'''
print(x_out.shape, y_out.shape)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
ax1.imshow(montage_nd(x_out[..., 0]), cmap='bone')
ax1.set_title('In Batch')
ax2.imshow(montage_nd(y_out[...,0]))
ax2.set_title('Out Batch')
plt.show()
'''
model = make_model()
# temp_pred = model.predict(x_out)
# print('temp_pred', temp_pred.shape)
# print('x_out', x_out.shape)
# temp_pred = model.predict(x_out)
# print('temp_pred', temp_pred.shape)
# for x, y in train_gen:
#    t_p = model.predict(x)
#    print(x.shape, y.shape, t_p.shape)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy', 'mse'])
weights_path = '{}_weights.best.hdf5'.format('convlstm_model')
checkpoint = tensorflow.keras.callbacks.ModelCheckpoint(weights_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=True)
reducelr = tensorflow.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=5, verbose=1, mode='auto', epsilon=1e-4, cooldown=5, min_lr=1e-4)
early = tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=10)
callbacks_list = [checkpoint, reducelr, early]
model.fit_generator(
    train_gen,
    epochs=2,
    steps_per_epoch=100,
    validation_data=valid_gen,
    validation_steps=10,
    callbacks=callbacks_list
)




