from keras.layers import Input, Conv2D, Lambda, Dense, Flatten, MaxPooling2D
from keras.models import Model, Sequential
from keras.initializers import RandomNormal
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
import numpy as np

class SiameseNet:
    def __init__(self, input_shape):
        self.__input_shape = input_shape
        self.model = self.__build()

    def __get_convolutional_layer(self, num_filters, kernel_size):
        convolutional_layer = Conv2D(
            num_filters, 
            kernel_size,
            activation='relu',
            kernel_initializer=RandomNormal(0, 1e-2),
            kernel_regularizer=l2(2e-4),
            bias_initializer=RandomNormal(0.5, 1e-2)
        )
        
        return convolutional_layer

    def __get_features_extractor(self, name='encoder'):
        features_extractor = Sequential(name=name)

        features_extractor.add(
            Conv2D(
                64,
                10,
                activation='relu',
                input_shape=self.__input_shape,
                kernel_initializer=RandomNormal(0, 1e-2),
                kernel_regularizer=l2(2e-4),
            )
        )
        features_extractor.add(MaxPooling2D())

        features_extractor.add(self.__get_convolutional_layer(128, 7))
        features_extractor.add(MaxPooling2D())
        features_extractor.add(self.__get_convolutional_layer(128, 4))
        features_extractor.add(MaxPooling2D())
        features_extractor.add(self.__get_convolutional_layer(256, 4))
        
        features_extractor.add(Flatten())
        features_extractor.add(
            Dense(
                4096, 
                activation="sigmoid", 
                kernel_regularizer=l2(1e-3), 
                kernel_initializer=RandomNormal(0, 1e-2),
                bias_initializer=RandomNormal(0.5, 1e-2)
            )
        )

        return features_extractor
    
    def __build(self):
        left_encoder_input = Input(self.__input_shape, name='left_encoder_input')
        right_encoder_input = Input(self.__input_shape, name='right_encoder_input')

        encoder = self.__get_features_extractor()
        left_encoder = encoder(left_encoder_input)
        right_encoder = encoder(right_encoder_input)

        L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]), name='l1_distance')
        L1_distance_layer = L1_layer([left_encoder, right_encoder])

        prediction = Dense(
            units=1, 
            activation='sigmoid', 
            bias_initializer=RandomNormal(0.5, 1e-2)
        )(L1_distance_layer)
        
        model = Model(inputs=[left_encoder_input, right_encoder_input], outputs= prediction, name='Siamese')

        optimizer = Adam()
        model.compile(loss='binary_crossentropy', optimizer=optimizer)

        return model