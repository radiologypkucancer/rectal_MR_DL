# --------------------------------------------------------------------------------------------------
# imports
import os

from keras import backend as K
from keras.models import Model
from keras.layers import concatenate
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from keras import regularizers
from keras.optimizers import Adam
from keras.utils import plot_model

# using keras wiht tensorflow as backend
# using the channel_first image data format
# all the input into keras net uses the settings of (channels_first) as default
#   2d: (samples, channels, height, width)
#   3d: (samples, channels, slices, height, width)

os.environ['KERAS_BACKEND'] = 'tensorflow'
K.set_image_data_format('channels_first')

# --------------------------------------------------------------------------------------------------
#
# DCPCNN: Densely Connected multi-max-Pooling Convolutional Neural Network
#
# --------------------------------------------------------------------------------------------------

class DCPCNN:
    # -----------------------------------------------------------------------------------------------
    #
    def __init__(self, input_img_size= (16, 256, 256), nb_classx=2, conv_depth=5, filters=16):
        # Input shape
        self.input_img_size = input_img_size
        # Output class number
        self.nb_classx = nb_classx
        # Network params
        self.kernel = (3, 3)
        self.pooling_size = (2, 2)
        self.conv_depth = conv_depth
        self.filters = filters
        self.bn_axis = 1

    # -----------------------------------------------------------------------------------------------
    #
    def initialize_model(self):
        #input and output shapes
        data_niix_size = self.input_img_size
        nb_classx = self.nb_classx

        # 8 paths for pre/post-NCRT T2W/Dapp/Kapp/Sapp images
        input_vol_1 = Input(data_niix_size)
        input_vol_2 = Input(data_niix_size)
        input_vol_3 = Input(data_niix_size)
        input_vol_4 = Input(data_niix_size)
        input_vol_5 = Input(data_niix_size)
        input_vol_6 = Input(data_niix_size)
        input_vol_7 = Input(data_niix_size)
        input_vol_8 = Input(data_niix_size)

        # extract features for each path
        x_tensor1 = self._feature_extraction_channel(input_vol_1)
        x_tensor2 = self._feature_extraction_channel(input_vol_2)
        x_tensor3 = self._feature_extraction_channel(input_vol_3)
        x_tensor4 = self._feature_extraction_channel(input_vol_4)
        x_tensor5 = self._feature_extraction_channel(input_vol_5)
        x_tensor6 = self._feature_extraction_channel(input_vol_6)
        x_tensor7 = self._feature_extraction_channel(input_vol_7)
        x_tensor8 = self._feature_extraction_channel(input_vol_8)

        # concatenate all 8 paths' features
        x_tensor = concatenate([x_tensor1, x_tensor2, x_tensor3, x_tensor4, x_tensor5, x_tensor6, x_tensor7, x_tensor8], axis=1)

        # fully connected layers
        x_tensor = Dense(nb_classx * 16, activation='relu', kernel_regularizer=regularizers.l2(0.025))(x_tensor)
        x_tensor = Dropout(0.5)(x_tensor)
        x_tensor = Dense(nb_classx * 4, activation='relu', kernel_regularizer=regularizers.l2(0.025))(x_tensor)

        # classfication
        output_tensor = Dense(nb_classx,activation='softmax')(x_tensor)


        # the multipath_net model
        self.multipath_net = Model(inputs=[input_vol_1, input_vol_2, input_vol_3, input_vol_4, input_vol_5, input_vol_6, input_vol_7, input_vol_8], outputs=output_tensor)
        self.multipath_net.compile(optimizer=Adam(lr=7.0e-6,decay=1.0e-5), loss='categorical_crossentropy', metrics=['mae', 'acc'])


        return True

    #-----------------------------------------------------------------------------------------------
    #
    def train(self, input_netsdata, classx_netsdata, batch_size, epochs, **kwargs):

        # add validation data
        validation_data = None
        validation_input_netsdata = kwargs.get('validation_input_netsdata')
        validation_target_netsddata = kwargs.get('validation_classx_netsdata')
        if validation_input_netsdata is not None and validation_target_netsddata is not None:
            validation_data = (validation_input_netsdata, validation_target_netsddata)

        self.multipath_net.fit(input_netsdata, \
                       classx_netsdata, \
                       batch_size=batch_size, \
                       epochs=epochs, \
                       verbose=2, \
                       shuffle=True, \
                       validation_data=validation_data,
                       validation_split=0.0)

    # -----------------------------------------------------------------------------------------------
    #
    def predict(self, input_netsdata, **kwargs):
        return self.multipath_net.predict(input_netsdata, verbose=0)

    # -----------------------------------------------------------------------------------------------
    #
    def _multi_pooling(self, pooling_size, input_conv_feature_map, input_raw_img):

        F0 = MaxPooling2D(pool_size=pooling_size, strides=2)(input_conv_feature_map)
        F1 = MaxPooling2D(pool_size=pooling_size, strides=2)(input_raw_img)
        multicrop_pool = concatenate([F0, F1], axis=1)
        return  multicrop_pool

    # --------------------------------------------------------------------------------
    #
    def _feature_extraction_channel(self, input_volume):
        data_niix_size = self.input_img_size
        kernel = self.kernel
        pooling_size= self.pooling_size
        conv_depth = self.conv_depth
        filters = self.filters
        bn_axis = self.bn_axis

        # blocks - n
        for block_i in range(conv_depth):
            conv = Conv2D(filters, kernel, activation='relu', kernel_regularizer=regularizers.l2(0.025), padding='same')(input_volume) # lamda2 = 0.025
            conv = BatchNormalization(axis=bn_axis)(conv)
            cropsize = int(data_niix_size[2]/(2**(2+block_i)))
            if cropsize < 2 : break
            # rectal MR model use _multi_pooling module
            input_volume = self._multi_pooling(pooling_size=pooling_size, input_conv_feature_map=conv, input_raw_img=input_volume)

        shape = input_volume.get_shape()
        kernel_size = (int(shape[2]), int(shape[3]))
        x_tensor = Conv2D(filters, kernel_size=kernel_size, activation='relu', kernel_regularizer=regularizers.l2(0.025))(input_volume)
        x_tensor = BatchNormalization(axis=bn_axis)(x_tensor)

        x_tensor = Flatten()(x_tensor)

        return x_tensor

    # --------------------------------------------------------------------------------
    # plot and show the model architecture
    def save_net_summary_to_file(self, file=''):
        plot_model(self.multipath_net, to_file=file, show_shapes=True)
        return True

    # --------------------------------------------------------------------------------
    # save weights
    def save_weights(self, file):
        self.multipath_net.save_weights(file)
        return True

    # --------------------------------------------------------------------------------
    # load weights
    def load_weights(self, file):
        self.multipath_net.load_weights(file)
        return True
