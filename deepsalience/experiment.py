"""common code for all experiments
"""
from __future__ import print_function

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

import numpy as np
np.random.seed(1337)
import keras
import medleydb as mdb
import os

import core
import evaluate

SAMPLES_PER_EPOCH = 512
NB_EPOCHS = 100
NB_VAL_SAMPLES = 512


def experiment(save_key, model):
    """common code for all experiments
    """
    exper_dir = core.experiment_output_path()
    data_path = core.data_path_multif0_complete()
    mtrack_list = core.track_id_list()
    input_patch_size = core.patch_size()

    (SAVE_PATH, MODEL_SAVE_PATH, PLOT_SAVE_PATH,
     MODEL_SCORES_PATH, SCORES_PATH, SCORE_SUMMARY_PATH
    ) = evaluate.get_paths(exper_dir, save_key)

    ### DATA SETUP ###
    dat = core.Data(
        mtrack_list, data_path, input_patch_size=input_patch_size
    )
    train_generator = dat.get_train_generator()
    validation_generator = dat.get_validation_generator()

    model.compile(
        loss=core.bkld, metrics=['mse', core.soft_binary_accuracy],
        optimizer='adam'
    )

    print(model.summary(line_length=80))

    ### FIT MODEL ###
    history = model.fit_generator(
        train_generator, SAMPLES_PER_EPOCH, epochs=NB_EPOCHS, verbose=1,
        validation_data=validation_generator, validation_steps=NB_VAL_SAMPLES,
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                MODEL_SAVE_PATH, save_best_only=True, verbose=1),
            keras.callbacks.ReduceLROnPlateau(patience=5, verbose=1),
            keras.callbacks.EarlyStopping(patience=15, verbose=0)
        ]
    )

    ### load best weights ###
    model.load_weights(MODEL_SAVE_PATH)

    ### Results plots ###
    print("plotting results...")
    evaluate.plot_metrics_epochs(history, PLOT_SAVE_PATH)

    ### Evaluate ###
    print("getting model metrics...")
    evaluate.get_model_metrics(dat, model, MODEL_SCORES_PATH)

    print("getting multif0 metrics...")
    evaluate.get_all_multif0_metrics(
        dat.test_files, model, SAVE_PATH, SCORES_PATH, SCORE_SUMMARY_PATH
    )

    bach10_files = core.get_file_paths(mdb.TRACK_LIST_BACH10, dat.data_path)
    evaluate.get_all_multif0_metrics(
        bach10_files, model,
        SAVE_PATH,
        os.path.join(SAVE_PATH, "bach10_scores.csv"),
        os.path.join(SAVE_PATH, "bach10_score_summary.csv"), create_pred=True)

    print("done!")
    print("Results saved to {}".format(SAVE_PATH))


