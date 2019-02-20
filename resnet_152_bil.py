# -*- coding: utf-8 -*-

from keras.optimizers import SGD
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Flatten, Activation, add
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import backend as K

from sklearn.metrics import log_loss

from custom_layers.scale_layer import Scale

import sys
sys.setrecursionlimit(3000)

import numpy as np
import tensorflow as tf
from keras import Input
from keras.layers import Conv2D, MaxPooling2D, merge, Lambda, Dense, Reshape, regularizers

 
def compact_bilinear(tensors_list):

    def _generate_sketch_matrix(rand_h, rand_s, output_dim):
        """
        Return a sparse matrix used for tensor sketch operation in compact bilinear
        pooling
        Args:
            rand_h: an 1D numpy array containing indices in interval `[0, output_dim)`.
            rand_s: an 1D numpy array of 1 and -1, having the same shape as `rand_h`.
            output_dim: the output dimensions of compact bilinear pooling.
        Returns:
            a sparse matrix of shape [input_dim, output_dim] for tensor sketch.
        """

        # Generate a sparse matrix for tensor count sketch
        rand_h = rand_h.astype(np.int64)
        rand_s = rand_s.astype(np.float32)
        assert (rand_h.ndim == 1 and rand_s.ndim == 1 and len(rand_h) == len(rand_s))
        assert (np.all(rand_h >= 0) and np.all(rand_h < output_dim))

        input_dim = len(rand_h)
        indices = np.concatenate((np.arange(input_dim)[..., np.newaxis],
                                  rand_h[..., np.newaxis]), axis=1)
        sparse_sketch_matrix = tf.sparse_reorder(
            tf.SparseTensor(indices, rand_s, [input_dim, output_dim]))
        return sparse_sketch_matrix

    bottom1, bottom2 = tensors_list
    output_dim = 8192

    # Static shapes are needed to construction count sketch matrix
    input_dim1 = bottom1.get_shape().as_list()[-1]
    input_dim2 = bottom2.get_shape().as_list()[-1]

    # print (bottom1.get_shape().as_list())
    # print (bottom2.get_shape().as_list())

    # Step 0: Generate vectors and sketch matrix for tensor count sketch
    # This is only done once during graph construction, and fixed during each
    # operation
    seed_h_1 = 1
    seed_s_1 = 3
    seed_h_2 = 5
    seed_s_2 = 7

    # Generate sparse_sketch_matrix1 using rand_h_1 and rand_s_1
    np.random.seed(seed_h_1)
    rand_h_1 = np.random.randint(output_dim, size=input_dim1)
    np.random.seed(seed_s_1)
    rand_s_1 = 2 * np.random.randint(2, size=input_dim1) - 1
    sparse_sketch_matrix1 = _generate_sketch_matrix(rand_h_1, rand_s_1, output_dim)

    # Generate sparse_sketch_matrix2 using rand_h_2 and rand_s_2
    np.random.seed(seed_h_2)
    rand_h_2 = np.random.randint(output_dim, size=input_dim2)
    np.random.seed(seed_s_2)
    rand_s_2 = 2 * np.random.randint(2, size=input_dim2) - 1
    sparse_sketch_matrix2 = _generate_sketch_matrix(rand_h_2, rand_s_2, output_dim)

    # Step 1: Flatten the input tensors and count sketch
    bottom1_flat = tf.reshape(bottom1, [-1, input_dim1])
    bottom2_flat = tf.reshape(bottom2, [-1, input_dim2])

    # Essentially:
    #   sketch1 = bottom1 * sparse_sketch_matrix
    #   sketch2 = bottom2 * sparse_sketch_matrix
    # But tensorflow only supports left multiplying a sparse matrix, so:
    #   sketch1 = (sparse_sketch_matrix.T * bottom1.T).T
    #   sketch2 = (sparse_sketch_matrix.T * bottom2.T).T
    sketch1 = tf.transpose(tf.sparse_tensor_dense_matmul(sparse_sketch_matrix1,
                                                         bottom1_flat, adjoint_a=True, adjoint_b=True))
    sketch2 = tf.transpose(tf.sparse_tensor_dense_matmul(sparse_sketch_matrix2,
                                                         bottom2_flat, adjoint_a=True, adjoint_b=True))

    # Step 2: FFT
    fft1 = tf.fft(tf.complex(real=sketch1, imag=tf.zeros_like(sketch1)))
    fft2 = tf.fft(tf.complex(real=sketch2, imag=tf.zeros_like(sketch2)))

    # Step 3: Elementwise product
    fft_product = tf.multiply(fft1, fft2)

    # Step 4: Inverse FFT and reshape back
    # Compute output shape dynamically: [batch_size, height, width, output_dim]
    cbp_flat = tf.real(tf.ifft(fft_product))

    output_shape = tf.add(tf.multiply(tf.shape(bottom1), [1, 1, 1, 0]),
                          [0, 0, 0, output_dim])
    cbp = tf.reshape(cbp_flat, output_shape)

    # print (cbp.get_shape().as_list())

    return cbp
def identity_block(input_tensor, kernel_size, filters, stage, block):
    '''The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    '''
    eps = 1.1e-5
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    scale_name_base = 'scale' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a', use_bias=False)(input_tensor)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2a')(x)
    x = Activation('relu', name=conv_name_base + '2a_relu')(x)

    x = ZeroPadding2D((1, 1), name=conv_name_base + '2b_zeropadding')(x)
    x = Conv2D(nb_filter2, (kernel_size, kernel_size),
                      name=conv_name_base + '2b', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2b')(x)
    x = Activation('relu', name=conv_name_base + '2b_relu')(x)

    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2c')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2c')(x)

    x = add([x, input_tensor], name='res' + str(stage) + block)
    x = Activation('relu', name='res' + str(stage) + block + '_relu')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    '''conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    '''
    eps = 1.1e-5
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    scale_name_base = 'scale' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, (1, 1), strides=strides,
                      name=conv_name_base + '2a', use_bias=False)(input_tensor)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2a')(x)
    x = Activation('relu', name=conv_name_base + '2a_relu')(x)

    x = ZeroPadding2D((1, 1), name=conv_name_base + '2b_zeropadding')(x)
    x = Conv2D(nb_filter2, (kernel_size, kernel_size),
                      name=conv_name_base + '2b', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2b')(x)
    x = Activation('relu', name=conv_name_base + '2b_relu')(x)

    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2c')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2c')(x)

    shortcut = Conv2D(nb_filter3, (1, 1), strides=strides,
                             name=conv_name_base + '1', use_bias=False)(input_tensor)
    shortcut = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '1')(shortcut)
    shortcut = Scale(axis=bn_axis, name=scale_name_base + '1')(shortcut)

    x = add([x, shortcut], name='res' + str(stage) + block)
    x = Activation('relu', name='res' + str(stage) + block + '_relu')(x)
    return x

def resnet152_model(img_rows, img_cols, color_type=1, num_classes=None):
    """
    Resnet 152 Model for Keras

    Model Schema and layer naming follow that of the original Caffe implementation
    https://github.com/KaimingHe/deep-residual-networks

    ImageNet Pretrained Weights 
    Theano: https://drive.google.com/file/d/0Byy2AcGyEVxfZHhUT3lWVWxRN28/view?usp=sharing
    TensorFlow: https://drive.google.com/file/d/0Byy2AcGyEVxfeXExMzNNOHpEODg/view?usp=sharing

    Parameters:
      img_rows, img_cols - resolution of inputs
      channel - 1 for grayscale, 3 for color 
      num_classes - number of class labels for our classification task
    """
    eps = 1.1e-5

    # Handle Dimension Ordering for different backends
    global bn_axis
    if K.image_dim_ordering() == 'tf':
      bn_axis = 3
      img_input = Input(shape=(img_rows, img_cols, color_type), name='data')
    else:
      bn_axis = 1
      img_input = Input(shape=(color_type, img_rows, img_cols), name='data')

    x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name='bn_conv1')(x)
    x = Scale(axis=bn_axis, name='scale_conv1')(x)
    x = Activation('relu', name='conv1_relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    for i in range(1,8):
      x = identity_block(x, 3, [128, 128, 512], stage=3, block='b'+str(i))

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    for i in range(1,36):
      x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b'+str(i))

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x_fc = AveragePooling2D((7, 7), name='avg_pool')(x)
    x_fc = Flatten()(x_fc)
    x_fc = Dense(1000, activation='softmax', name='fc1000')(x_fc)

    model = Model(img_input, x_fc)

    if K.image_dim_ordering() == 'th':
      # Use pre-trained weights for Theano backend
      weights_path = 'models/resnet152_weights_th.h5'
    else:
      # Use pre-trained weights for Tensorflow backend
      weights_path = 'models/resnet152_weights_tf.h5'

    model.load_weights(weights_path, by_name=True)

    # Truncate and replace softmax layer for transfer learning
    # Cannot use model.layers.pop() since model is not of Sequential() type
    # The method below works since pre-trained weights are stored in layers but not in the model
    compact_bilinear_arg_list = [x,x]

    output_shape_x = x.get_shape().as_list()[1:]
    output_shape_cb = (output_shape_x[0], output_shape_x[1], 8192,)
    #x = merge(compact_bilinear_arg_list, mode=compact_bilinear, name='compact_bilinear', output_shape=output_shape_cb)
    x = Lambda(compact_bilinear,output_shape_cb)(compact_bilinear_arg_list)
    # Sign sqrt and L2 normalize result
    x = Lambda(lambda x: K.sign(x) * K.sqrt(K.abs(x)))(x)
    x = Lambda(lambda x: K.l2_normalize(x, axis=-1))(x)
    x_newfc = Flatten()(x)
    x_newfc = Dense(num_classes, activation='softmax', name='fc8')(x_newfc)

    model = Model(img_input, x_newfc)

    # Learning rate is changed to 0.001
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

if __name__ == '__main__':

    # Example to fine-tune on 3000 samples from Cifar10

    img_rows, img_cols = 224, 224 # Resolution of inputs
    channel = 3
    num_classes = 10 
    batch_size = 8
    epochs = 10

    # Load Cifar10 data. Please implement your own load_data() module for your own dataset
    X_train, Y_train, X_valid, Y_valid = load_cifar10_data(img_rows, img_cols)

    # Load our model
    model = resnet152_model(img_rows, img_cols, channel, num_classes)

    # Start Fine-tuning
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=epochs,
              shuffle=True,
              verbose=1,
              validation_data=(X_valid, Y_valid),
              )

    # Make predictions
    predictions_valid = model.predict(X_valid, batch_size=batch_size, verbose=1)

    # Cross-entropy loss score
    score = log_loss(Y_valid, predictions_valid)
