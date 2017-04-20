from __future__ import print_function
import keras
from keras.models import Model
from keras.layers import Dense, Input, Reshape, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K
import os

import experiment

def model_def():
    ### DEFINE MODEL ###
    input_shape = (None, None, 6)
    inputs = Input(shape=input_shape)

    y0 = BatchNormalization()(inputs)
    y1 = Conv2D(128, (5, 5), padding='same', activation='relu', name='bendy1')(y0)
    y1a = BatchNormalization()(y1)
    y2 = Conv2D(64, (5, 5), padding='same', activation='relu', name='bendy2')(y1a)
    y2a = BatchNormalization()(y2)
    y3 = Conv2D(64, (3, 3), padding='same', activation='relu', name='smoothy1')(y2a)
    y3a = BatchNormalization()(y3)
    y4 = Conv2D(64, (3, 3), padding='same', activation='relu', name='smoothy2')(y3a)
    y4a = BatchNormalization()(y4)
    y5 = Conv2D(8, (70, 3), padding='same', activation='relu', name='harm')(y4a)
    y5a = BatchNormalization()(y5)
    y6 = Conv2D(8, (360, 1), padding='same', activation='relu', name='distribute')(y5a)
    y6a = BatchNormalization()(y6)
    y7 = Conv2D(1, (1, 1), padding='same', activation='sigmoid', name='squishy')(y6a)
    predictions = Lambda(lambda x: K.squeeze(x, axis=3))(y7)

    model = Model(inputs=inputs, outputs=predictions)
    return model


def main():

    save_key = os.path.basename(__file__).split('.')[0]
    model = model_def()
    experiment.experiment(save_key, model)

if __name__ == '__main__':
    main()
