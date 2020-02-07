# https://github.com/ardamavi/3D-Medical-Segmentation-GAN/blob/master/get_models.py
import os
import keras
import keras.backend as k


def dice_coefficient(y_true, y_pred):
    smoothing = 1.0
    flat_y_true = k.flatten(y_true)
    flat_y_pred = k.flatten(y_pred)
    return (2.0 * k.sum(flat_y_true * flat_y_pred) + smoothing) / (k.sum(flat_y_true) + k.sum(flat_y_pred) + smoothing)


def dice_coefficient_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)


def get_segment_model(data_shape):
    inputs = keras.layers.Input(shape=(data_shape))

    conv_block_1 = keras.layers.Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(inputs)
    conv_block_1 = keras.layers.Activation('relu')(conv_block_1)
    conv_block_1 = keras.layers.Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(
        conv_block_1)
    conv_block_1 = keras.layers.Activation('relu')(conv_block_1)
    conv_block_1 = keras.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(conv_block_1)

    conv_block_2 = keras.layers.Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(
        conv_block_1)
    conv_block_2 = keras.layers.Activation('relu')(conv_block_2)
    conv_block_2 = keras.layers.Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(
        conv_block_2)
    conv_block_2 = keras.layers.Activation('relu')(conv_block_2)
    conv_block_2 = keras.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(conv_block_2)

    conv_block_3 = keras.layers.Conv3D(filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(
        conv_block_2)
    conv_block_3 = keras.layers.Activation('relu')(conv_block_3)
    conv_block_3 = keras.layers.Conv3D(filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(
        conv_block_3)
    conv_block_3 = keras.layers.Activation('relu')(conv_block_3)
    conv_block_3 = keras.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(conv_block_3)

    conv_block_4 = keras.layers.Conv3D(filters=256, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(
        conv_block_3)
    conv_block_4 = keras.layers.Activation('relu')(conv_block_4)
    conv_block_4 = keras.layers.Conv3D(filters=256, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(
        conv_block_4)
    conv_block_4 = keras.layers.Activation('relu')(conv_block_4)
    conv_block_4 = keras.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(conv_block_4)

    conv_block_5 = keras.layers.Conv3D(filters=256, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(conv_block_4)
    conv_block_5 = keras.layers.Activation('relu')(conv_block_5)
    conv_block_5 = keras.layers.Conv3D(filters=256, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(conv_block_5)
    conv_block_5 = keras.layers.Activation('relu')(conv_block_5)

    encoder = keras.Model(inputs=inputs, outputs=conv_block_5)

    up_block_1 = keras.layers.UpSampling3D((2, 2, 2))(conv_block_5)
    up_block_1 = keras.layers.Conv3D(filters=512, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(up_block_1)

    merge_1 = keras.layers.concatenate([conv_block_4, up_block_1])

    conv_block_6 = keras.layers.Conv3D(filters=256, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(merge_1)
    conv_block_6 = keras.layers.Activation('relu')(conv_block_6)
    conv_block_6 = keras.layers.Conv3D(filters=256, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(conv_block_6)
    conv_block_6 = keras.layers.Activation('relu')(conv_block_6)

    up_block_2 = keras.layers.UpSampling3D((2, 2, 2))(conv_block_6)
    up_block_2 = keras.layers.Conv3D(filters=256, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(up_block_2)

    merge_2 = keras.layers.concatenate([conv_block_3, up_block_2])

    conv_block_7 = keras.layers.Conv3D(filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(merge_2)
    conv_block_7 = keras.layers.Activation('relu')(conv_block_7)
    conv_block_7 = keras.layers.Conv3D(filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(conv_block_7)
    conv_block_7 = keras.layers.Activation('relu')(conv_block_7)

    up_block_3 = keras.layers.UpSampling3D((2, 2, 2))(conv_block_7)
    up_block_3 = keras.layers.Conv3D(filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(up_block_3)

    merge_3 = keras.layers.concatenate([conv_block_2, up_block_3])

    conv_block_8 = keras.layers.Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(merge_3)
    conv_block_8 = keras.layers.Activation('relu')(conv_block_8)
    conv_block_8 = keras.layers.Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(conv_block_8)
    conv_block_8 = keras.layers.Activation('relu')(conv_block_8)

    up_block_4 = keras.layers.UpSampling3D((2, 2, 2))(conv_block_8)
    up_block_4 = keras.layers.Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(up_block_4)

    merge_4 = keras.layers.concatenate([conv_block_1, up_block_4])

    conv_block_9 = keras.layers.Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(merge_4)
    conv_block_9 = keras.layers.Activation('relu')(conv_block_9)
    conv_block_9 = keras.layers.Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(conv_block_9)
    conv_block_9 = keras.layers.Activation('relu')(conv_block_9)

    conv_block_10 = keras.layers.Conv3D(filters=data_shape[-1], kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same')(conv_block_9)
    outputs = keras.layers.Activation('sigmoid')(conv_block_10)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adadelta(lr=1e-02), loss=dice_coefficient_loss, metrics=[dice_coefficient, 'acc'])
    return model, encoder


def get_GAN(input_shape, Generator, Discriminator):
    input_gan = keras.layers.Input(shape=input_shape)
    generated_seg = Generator(input_gan)
    gan_output = Discriminator([input_gan, generated_seg])
    gan = keras.Model(input_gan, gan_output)
    gan.compile(optimizer=keras.optimizers.Adadelta(lr=1e-02), loss='mse', metrics=['accuracy'])
    return gan


def get_Generator(input_shape):
    Generator, _ = get_segment_model(input_shape)
    Generator.summary()
    return Generator


def get_Discriminator(input_shape_1, input_shape_2, Encoder):
    dis_input_1 = keras.layers.Input(shape=input_shape_1)
    dis_input_2 = keras.layers.Input(shape=input_shape_2)
    mul_1 = keras.layers.Multiply()([dis_input_1, dis_input_2])
    encoder_output_1 = Encoder(dis_input_1)
    encoder_output_2 = Encoder(mul_1)
    subtract_dis = keras.layers.Subtract()([encoder_output_1, encoder_output_2])

    dis_conv_block = keras.layers.Conv3D(filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(subtract_dis)
    dis_conv_block = keras.layers.Activation('relu')(dis_conv_block)
    dis_conv_block = keras.layers.Conv3D(filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(dis_conv_block)
    dis_conv_block = keras.layers.Activation('relu')(dis_conv_block)
    dis_conv_block = keras.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(dis_conv_block)

    dis_conv_block = keras.layers.Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(dis_conv_block)
    dis_conv_block = keras.layers.Activation('relu')(dis_conv_block)
    dis_conv_block = keras.layers.Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(dis_conv_block)
    dis_conv_block = keras.layers.Activation('relu')(dis_conv_block)

    dis_conv_block = keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(dis_conv_block)
    dis_conv_block = keras.layers.Activation('relu')(dis_conv_block)
    dis_conv_block = keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(dis_conv_block)
    dis_conv_block = keras.layers.Activation('relu')(dis_conv_block)

    flat_1 = keras.layers.Flatten()(dis_conv_block)
    dis_fc_1 = keras.layers.Dense(256)(flat_1)
    dis_fc_1 = keras.layers.Activation('relu')(dis_fc_1)
    dis_drp_1 = keras.layers.Dropout(0.5)(dis_fc_1)
    dis_fc_2 = keras.layers.Dense(128)(dis_drp_1)
    dis_fc_2 = keras.layers.Activation('relu')(dis_fc_2)
    dis_drp_2 = keras.layers.Dropout(0.5)(dis_fc_2)
    dis_fc_3 = keras.layers.Dense(1)(dis_drp_2)
    dis_similarity_output = keras.layers.Activation('sigmoid')(dis_fc_3)

    Discriminator = keras.Model(inputs=[dis_input_1, dis_input_2], outputs=dis_similarity_output)
    Discriminator.compile(optimizer=keras.optimizers.Adadelta(lr=1e-02), loss='binary_crossentropy', metrics=['accuracy'])
    Discriminator.summary()
    return Discriminator

