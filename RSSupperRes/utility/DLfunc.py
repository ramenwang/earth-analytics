import numpy as np
import tensorflow as tf
import keras as K
from keras.layers import Input, Concatenate, Conv2D, ReLU, Dense, Lambda
from keras.layers import BatchNormalization, Dropout
from keras.optimizers import Adam
from keras.models import Model


def conv2d(x, num_filters, filter_size, stride=(1,1), padding='same', name="",
           act=True, batchnorm=True, dropout_rate=None):
    ''' conv2d with optional batchnorm, ReLU activation, or dropout
    :param x: input tensor
    :param num_filters: number of filters
    :param filter_size: filter size
    :param stride: convolution stride
    :param padding: padding
    :param name: give name to the convolutional operation
    :param act: if giving activation
    :param batchnorm: if implement batch normalization
    :param dropout_rate: if None no random dropout, otherwise random dropout based on the rate
    :return: a tensor after convolutional operation
    :rtype: tensor
    '''
    x = Conv2D(filters=num_filters, kernel_size=filter_size, strides=stride,
               padding=padding, name=name+'_conv2d')(x)
    # batch normalization
    if (batchnorm):
        x = BatchNormalization()(x)
    # activation
    if (act):
        x = ReLU()(x)
    # random dropout
    if (dropout_rate != None):
        x = Dropout(rate=dropout_rate)(x)

    return x


def inception_module(x, nin_filter, dropout_rate,batchnorm=True, name=""):
    ''' inception layer with bottleneck layer to reduce the depth
    :param nin_filter1: number of filters used on inception path 1
    :param nin_filter2: number of filters used on inception path 2
    :param dropout_rate: if None no random dropout, otherwise random dropout based on the rate
    :param name: give name to the convolutional operation
    :param batchnorm: if implement batch normalization
    :return: tensor after inceptional feature selection
    :rtype: tensor
    '''
    # path 1 - bottleneck layer
    p1 = conv2d(x, num_filters=nin_filter, filter_size=(1,1), name=name+'_p1_bn',
                act=True, batchnorm=batchnorm, dropout_rate=dropout_rate)
    # path 2 - bottleneck layer + 3*3 filter
    nin_filter2 = nin_filter // 2
    p2 = conv2d(x, num_filters=nin_filter2, filter_size=(1,1), name=name+'_p2_bn',
                act=True, batchnorm=batchnorm, dropout_rate=dropout_rate)
    p2 = conv2d(p2, num_filters=nin_filter2, filter_size=(3,3), name=name+'_p2',
                act=True, batchnorm=batchnorm, dropout_rate=dropout_rate)
    # path 3 - bottleneck layer + 3*3 filter
    p3 = conv2d(x, num_filters=nin_filter2, filter_size=(1,1), name=name+'_p3_bn',
                act=True, batchnorm=batchnorm, dropout_rate=dropout_rate)
    p3 = conv2d(p3, num_filters=nin_filter2, filter_size=(5,5), name=name+'_p3',
                act=True, batchnorm=batchnorm, dropout_rate=dropout_rate)

    output = K.layers.concatenate([p1, p2, p3], axis=3, name=name+'_concat')
    return output


def pixel_shuffler(x, scale, output_filters, filter_size):
    ''' pixel shuffler layer to reconstruct the 2d space from feature map channels
    :param x: tensor object
    :param scale: scale factor to target image spatial resolution
    :param output_filters: number of the channels output from shuffler
    :param filter_size: the size of filter
    :return: a tensor which has the target spatial resolution and channels
    :rtype: activated tensor
    '''
    ps = conv2d(x, num_filters=output_filters*scale*scale, filter_size=filter_size,
                name='ps', batchnorm=False, act=False, dropout_rate=None)
    # using depth_to_space to resampling the image pixel from depth
    Subpixel_layer = Lambda(lambda x: tf.nn.depth_to_space(x,scale))
    ps = Subpixel_layer(inputs=ps)
    ps = LeakyReLU()(ps)
    return ps


def feature_extractor(x, n_layers, filter_size, n_filters, min_filters,
                      filter_decay_gamma, dropout_rate):
    ''' building feature extractor with skip connection and filter gamma decay
    :param x: input volumn
    :param n_layers: number of convolutional layers for feature extraction
    :param filter_size: size of the convolution filters
    :param n_filters: number of filters at the first layer
    :param min_filter: the minimum number of filters, normally the deepest layer
    :param filter_decay_gamma: the hyperparameter controls the number of filters
    :param dropout_rate: the rate of random dropout
    :return: a stack of skip connected tensors
    :rtype: tensors
    '''
    concat_list = []
    n_filter_list = []
    for i in range(n_layers):
        # perform gamma decay
        if min_filters !=0 and i > 0:
            x1 = i / float(n_layers - 1)
            y1 = pow(x1, 1. / filter_decay_gamma)
            n_filters = int((n_filters - min_filters) * (1 - y1) + min_filters)
            n_filter_list.append(n_filters)
        # perform convolution
        x = conv2d(x, num_filters=n_filters, filter_size=filter_size, name='FE_'+str(i),
                   act=True, batchnorm=True, dropout_rate=dropout_rate)
        concat_list.append(x)

    output = K.layers.concatenate(concat_list, axis=3, name='FE_concat')
    return output, n_filter_list


def get_psnr_tensor(x, mse):
    ''' compute PSNR for from MSE
    :param x: input tensor
    :param mse: mean square error
    :retun: PSNR value
    :rtype: mes.dtype
    '''
    value = tf.constant(tf.math.reduce_max(x), dtype=mse.dtype) / tf.sqrt(mse)
    numerator = tf.log(value)
    denominator = tf.log(tf.constant(10, dtype=mse.dtype))
    return tf.constant(20, dtype=mse.dtype) * numerator / denominator


def img_loss(border_size):
    ''' mean square error for tensor
    :param img_true: true image
    :param img_pred: predicted rescaled image
    :param border_size: according to the study, it is set to 2*filter_size+scale
    :return: mean sqaure error
    :rtype: constant
    '''
    def loss(img_true, img_pred):
        diff = K.layers.subtract([img_true, img_pred])
        if border_size>0:
            # trim image into wanted border_size
            diff = diff[border_size: -border_size, border_size: -border_size, -1]
        return K.backend.mean(K.backend.square(diff))

    return loss


def RSSuperRes(input_dim, scale, n_layers, n_filters, min_filters,
               filter_decay_gamma, dropout_rate, nin_filter,
               shuffler_output_filters, shuffler_filter_size,
               n_reconstructors, num_recon_filters, recon_filter_size,
               output_channels):
    ''' residual super resolution model, which includes four parts - feature extractor,
    inception feature selector, resolution resampler.
    :param x: input residual image between bicubic resampled image and target
    :param bicubic_x: input bicubic resampled image
    :param n_layers: number of layers in the feature extractor
    :param n_filters: number of filters in the first layer in feature extractor
    :param min_filters: number of filters in the deepest layer in feature extractor
    :param filter_decay_gamma: the hyperparameter controls the number of filters
    :param dropout_rate: the rate of random dropout
    :param nin_filter: number of filters used in inception module
    :param n_reconstructors: number of convolutional layers for resampling target
    :param num_recon_filters: number of filters in reconstructor
    :param output_channels: number of channels in target image
    :return: target image with higher resolution
    :rtype: tensor
    '''
    inputs = Input(shape=input_dim)
    x, n_filter_list = feature_extractor(inputs, n_layers, (3,3), n_filters,
                                         min_filters, filter_decay_gamma, dropout_rate)
    print("number of filters used in different layers are:")
    print(n_filter_list)
    x = inception_module(x=x, nin_filter=nin_filter, dropout_rate=dropout_rate, name='inc')
    x = pixel_shuffler(x=x, scale=scale, output_filters=shuffler_output_filters,
                       filter_size=shuffler_filter_size)
    for i in range(max(1, n_reconstructors)):
        x = conv2d(x=x, num_filters=num_recon_filters, filter_size=recon_filter_size,
                   batchnorm=False, dropout_rate=dropout_rate, name='recon'+str(i))
    outputs = conv2d(x=x, num_filters=output_channels, filter_size=(1,1), batchnorm=False,
                     dropout_rate=dropout_rate, act=False, name='output')

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
    model_name = str(scale) + "x." + str(n_layers) + 'lyr-FE-(' + str(n_layers) + '-' + \
                 '-'.join([str(i) for i in n_filter_list]) + \
                 ').3p-INC-(' + str(nin_filter) + '-' + str(nin_filter // 2) + ').' + \
                 str(n_reconstructors) + 'lyr-RCN-(' + str(num_recon_filters) + ').h5'

    return model, model_name
