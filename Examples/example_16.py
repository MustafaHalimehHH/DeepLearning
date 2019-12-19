# https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/model.py
import os
import random
import datetime
import re
import math
import logging
import multiprocessing
from collections import OrderedDict
import keras
import numpy
import tensorflow


def log(text, array=None):
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20} ".format(str(array.shape)))
        if array.size:
            text += ('min: {:10.5f}  max: {:10.5f}'.format(array.min(), array.max()))
        else:
            text += ('min: {:10} max: {:10}'.format("", ""))
        text += " {}".format(array.dtype)
    print(text)


class BatchNorm(keras.layers.BatchNormalization):
    def call(self, inputs, training=None):
        return super(self.__class__, self).call(inputs, training=training)


def compute_backbone_shapes(config, image_shape):
    if callable(config.BACKBONE):
        return config.COMPUTE_BACKBONE_SHAPE(image_shape)
    assert config.BACKBONE in ['resnet50', 'resnet101']
    return numpy.array([[
        int(math.ceil(image_shape[0] / stride)),
        int(math.ceil(image_shape[1] / stride))]
        for stride in config.BACKBONE_STRIDES
    ])


def identity_block(input_tensor, kernel_size, filters, stage, block, use_bias=True, train_bn=True):
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    x = keras.layers.Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
    x = BatchNorm(name=bn_name_base + '2a')(x, training=train_bn)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2c')(x, training=train_bn)
    x = keras.layers.Add()([x, input_tensor])
    x = keras.layers.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), use_bias=True, train_bn=True):
    nb_filer1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    x = keras.layers.Conv2D(nb_filer1, (1, 1), strides=strides, name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
    x = BatchNorm(name=bn_name_base + '2a')(x, training=train_bn)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2c')(x, training=train_bn)
    shortcut = keras.layers.Conv2D(nb_filter3, (1, 1), strides=strides, name=conv_name_base + '1', use_bias=use_bias)(input_tensor)
    shortcut = BatchNorm(name=bn_name_base + '1')(x, training=train_bn)
    x = keras.layers.Add([x, shortcut])
    x = keras.layers.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def resnet_graph(input_image, architecture, stage5=False, train_bn=True):
    assert architecture in ['resnet50', 'resnet101']
    x = keras.layers.ZeroPadding2D((3, 3))(input_image)
    x = keras.layers.Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True)(x)
    x = BatchNorm(name='bn_conv1')(x, training=train_bn)
    x = keras.layers.Activation('relu')(x)
    C1 = x = keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = conv_block(x, 3, [64, 64, 64], stage=2, block='a', strides=(1, 1), train_bn=train_bn)
    x = identity_block(x, 3, [64, 64, 64], stage=2, block='b', train_bn=train_bn)
    C2 = x = identity_block(x, 3, [64, 64, 64], stage=2, block='c', train_bn=train_bn)
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', train_bn=train_bn)
    C3 = x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', train_bn=train_bn)
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', train_bn=train_bn)
    block_count = {'resnet50': 5, 'resnet101': 22}[architecture]
    for i in block_count:
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i), train_bn=train_bn)
    C4 = x
    if stage5:
        x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', train_bn=train_bn)
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', train_bn=train_bn)
        C5 = x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', train_bn=train_bn)
    else:
        C5 = None
    return [C1, C2, C3, C4, C5]


def apply_box_deltas_graph(boxes, deltas):
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= tensorflow.exp(deltas[:, 2])
    width *= tensorflow.exp(deltas[:, 3])
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = tensorflow.stack([y1, x1, y2, x2], axis=1, name='apply_box_delta_out')
    return result


def clip_boxes_graph(boxes, window):
    wy1, wx1, wy2, wx2 = tensorflow.split(window, 4)
    y1, x1, y2, x2 = tensorflow.split(boxes, 4, axis=1)
    y1 = tensorflow.maximum(tensorflow.minimum(y1, wy2), wy1)
    x1 = tensorflow.maximum(tensorflow.minimum(x1, wx2), wx1)
    y2 = tensorflow.maximum(tensorflow.minimum(y2, wy2), wy1)
    x2 = tensorflow.maximum(tensorflow.minimum(x2, wx2), wx1)
    clipped = tensorflow.concat([y1, x1, y2, x2], axis=1, name='clipped_boxes')
    clipped.set_shape((clipped.shape[0], 4))
    return clipped


class ProposalLayer(keras.layers.Layer):
    def __init__(self, proposal_counts, nms_threshold, config=None, **kwargs):
        super(ProposalLayer, self).__init__(**kwargs)
        self.config = config
        self.proposal_counts = proposal_counts
        self.nms_threshold = nms_threshold


    def call(self, inputs):
        scores = inputs[0][:, :, 1]
        deltas = inputs[1]
        deltas = deltas * numpy.reshape(self.config.RPN_BBOX_STD_DEV, [1, 1, 4])
        anchors = inputs[2]
        pre_nms_limit = tensorflow.minimum(self.config.PRE_NMS_LIMIT, tensorflow.shape(anchors)[1])
        ix = tensorflow.nn.top_k(scores, pre_nms_limit, sorted=True, name='top_anchors').indices

