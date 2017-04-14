from __future__ import print_function

import keras
from keras.models import Model
from keras.layers import Dense, Input, Reshape, Lambda
from keras.layers.convolutional import Conv2D
from keras import backend as K

from tensorflow.python.client import device_lib
device_lib.list_local_devices() 

import medleydb as mdb
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas

np.random.seed(1337)  # for reproducibility

RANDOM_STATE = 42

import core

DATA_PATH = "/scratch/rmb456/multif0_ismir2017/training_data_with_blur/multif0_complete/"
MTRACK_LIST = mdb.TRACK_LIST_V1 + mdb.TRACK_LIST_V2 + mdb.TRACK_LIST_EXTRA + mdb.TRACK_LIST_BACH10
INPUT_PATCH_SIZE = (360, 50)
OUTPUT_PATH_SIZE = (360, 50)

SAMPLES_PER_EPOCH = 512
NB_EPOCHS = 100
NB_VAL_SAMPLES = 500


def main():

    SAVE_KEY = os.path.basename(__file__).split('.')[0]
    (SAVE_PATH, MODEL_SAVE_PATH, PLOT_SAVE_PATH,
     MODEL_SCORES_PATH, SCORES_PATH, SCORE_SUMMARY_PATH
    ) = core.get_paths("/home/rmb456/repos/multif0/experiment_output", SAVE_KEY)

    ### DATA SETUP ###
    dat = core.Data(
        MTRACK_LIST, DATA_PATH, input_patch_size=INPUT_PATCH_SIZE,
        output_patch_size=OUTPUT_PATH_SIZE, batch_size=10
    )
    train_generator = dat.get_train_generator()
    validation_generator = dat.get_validation_generator()

    ### DEFINE MODEL ###
    input_shape = (None, None, 6)
    inputs = Input(shape=input_shape)

    y1 = Conv2D(256, (5, 5), padding='same', activation='relu', name='bendy1')(inputs)
    y2 = Conv2D(64, (5, 5), padding='same', activation='relu', name='bendy2')(y1)
    y3 = Conv2D(64, (3, 3), padding='same', activation='relu', name='smoothy1')(y2)
    y4 = Conv2D(64, (3, 3), padding='same', activation='relu', name='smoothy2')(y3)
    y5 = Conv2D(1, (1, 1), padding='same', activation='sigmoid', name='squishy')(y4)
    predictions = Lambda(lambda x: K.squeeze(x, axis=3))(y5)

    model = Model(inputs=inputs, outputs=predictions)
    model.compile(
        loss=core.keras_loss(), metrics=core.keras_metrics(), optimizer='adam')

    print(model.summary(line_length=80))

    ### FIT MODEL ###
    history = model.fit_generator(
        train_generator, SAMPLES_PER_EPOCH, epochs=NB_EPOCHS, verbose=1,
        validation_data=validation_generator, validation_steps=NB_VAL_SAMPLES,
        callbacks=[
            keras.callbacks.ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, verbose=1),
            keras.callbacks.ReduceLROnPlateau(patience=5, verbose=1),
            keras.callbacks.EarlyStopping(patience=15, verbose=0)
        ]
    )

    ### load best weights ###
    model.load_weights(MODEL_SAVE_PATH)

    ### Results plots ###
    print("plotting results...")
    core.plot_metrics_epochs(history, PLOT_SAVE_PATH)

    ### Evaluate ###
    print("getting model metrics...")
    core.get_model_metrics(dat, model, MODEL_SCORES_PATH)

    print("getting multif0 metrics...")
    core.get_all_multif0_metrics(dat.test_files, model, SAVE_PATH, SCORES_PATH, SCORE_SUMMARY_PATH)

    print("done!")
    print("Results saved to {}".format(SAVE_PATH))

if __name__ == '__main__':
    main()
