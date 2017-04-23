from __future__ import print_function
import os
import glob
import importlib

import evaluate
import core

import keras
from keras.models import Model
from keras.layers import Dense, Input, Reshape, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K
import os
import numpy as np
import mir_eval
import traceback


def main():
    all_models = glob.glob(
        os.path.join("../experiment_output/multif0*/", "*.pkl")
    )
    for mpath in sorted(all_models):
        print("Running evaluation for {}".format(os.path.basename(mpath)))
        try:
            model_key = os.path.basename(mpath).split('.')[0]
            exper_module = importlib.import_module(model_key)
            save_path = os.path.dirname(mpath)

            # if the last scores file already exists, go to next model
            if os.path.exists(os.path.join(save_path, 'su_all_scores.csv')):
                print("Already done with model {}".format(mpath))
                continue

            # define model from module's model_def
            model = exper_module.model_def()
            # load the pretrained model
            model.load_weights(mpath)

            # load common variables
            data_path = core.data_path_multif0_complete()
            mtrack_list = core.track_id_list()
            input_patch_size = core.patch_size()
            dat = core.Data(
                mtrack_list, data_path, input_patch_size=input_patch_size
            )

            print("getting best threshold...")
            thresh = evaluate.get_best_thresh(dat, model)

            print("scoring multif0 metrics on test sets...")
            print("    > bach10...")
            evaluate.score_on_test_set('bach10', model, save_path, thresh)
            print("    > medleydb test...")
            evaluate.score_on_test_set('mdb_test', model, save_path, thresh)
            print("    > su...")
            evaluate.score_on_test_set('su', model, save_path, thresh)

        except:
            traceback.print_exc()


if __name__ == '__main__':
    main()
