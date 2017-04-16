"""Core data and model architecture classes
"""
from __future__ import print_function

import numpy as np
np.random.seed(1337)
import keras
from keras import backend as K
import pescador
import glob
import medleydb as mdb
from medleydb import utils
import os

RANDOM_STATE = 42


def patch_size():
    """Patch size used by all models for training
    """
    return (360, 50)


def experiment_output_path():
    return "/home/rmb456/repos/multif0/experiment_output"


def data_path_multif0_complete():
    """Data path for complete mulif0 data
    """
    return "/scratch/rmb456/multif0_ismir2017/" + \
        "training_data_with_blur/multif0_complete"


def data_path_multif0_incomplete():
    """Data path for incomplete multif0 data
    """
    return "/scratch/rmb456/multif0_ismir2017/" + \
        "training_data_with_blur/multif0_incomplete"


def track_id_list():
    """MedleyDB track ids used for train test and validation
    """
    return mdb.TRACK_LIST_V1 + mdb.TRACK_LIST_V2 + mdb.TRACK_LIST_EXTRA


def keras_loss():
    """Loss function used by all models
    """
    return bkld


def keras_metrics():
    """Metrics used by all models
    """
    return ['mse', soft_binary_accuracy]


def bkld(y_true, y_pred):
    """Brian's KL Divergence implementation
    """
    y_true = K.clip(y_true, K.epsilon(), 1.0 - K.epsilon())
    y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
    return K.mean(K.mean(
        -1.0*y_true* K.log(y_pred) - (1.0 - y_true) * K.log(1.0 - y_pred),
        axis=-1), axis=-1)


def soft_binary_accuracy(y_true, y_pred):
    """Binary accuracy that works when inputs are probabilities
    """
    return K.mean(K.mean(
        K.equal(K.round(y_true), K.round(y_pred)), axis=-1), axis=-1)


def keras_generator(data_list, input_patch_size):
    """Generator to be passed to a keras model
    """
    streams = []
    for fpath_in, fpath_out in data_list:
        streams.append(
            pescador.Streamer(
                patch_generator, fpath_in, fpath_out,
                input_patch_size=input_patch_size
            )
        )

    stream_mux = pescador.Mux(
        streams, 10, with_replacement=True, lam=500,
        random_state=RANDOM_STATE
    )

    for batch in stream_mux.tuples('X', 'Y'):
        yield batch


def grab_patch_output(f, t, n_f, n_t, y_data):
    """Get a time-frequency patch from an output file
    """
    return y_data[f: f + n_f, t: t + n_t][np.newaxis, :, :]


def grab_patch_input(f, t, n_f, n_t, x_data):
    """Get a time-frequency patch from an input file
    """
    return np.transpose(
        x_data[:, f: f + n_f, t: t + n_t], (1, 2, 0)
    )[np.newaxis, :, :, :]


def patch_generator(fpath_in, fpath_out, input_patch_size):
    """Generator that yields an infinite number of patches
       for a single input, output pair
    """
    data_in = np.load(fpath_in)
    data_out = np.load(fpath_out)

    _, _, n_times = data_in.shape
    n_f, n_t = input_patch_size

    t_vals = np.arange(0, n_times - n_t)
    np.random.shuffle(t_vals)

    for t in t_vals:
        f = 0
        t = np.random.randint(0, n_times - n_t)

        x = grab_patch_input(
            f, t, n_f, n_t, data_in
        )
        y = grab_patch_output(
            f, t, n_f, n_t, data_out
        )
        yield dict(X=x, Y=y)


def get_file_paths(mtrack_list, data_path):
    """Get the absolute paths to input/output pairs for
       a list of multitracks given a data path
    """
    file_paths = []
    for track_id in mtrack_list:
        input_path = glob.glob(
            os.path.join(data_path, 'inputs', "{}*_input.npy".format(track_id))
        )
        output_path = glob.glob(
            os.path.join(
                data_path, 'outputs', "{}*_output.npy".format(track_id)
            )
        )

        if len(input_path) == 1 and len(output_path) == 1:
            input_path = input_path[0]
            output_path = output_path[0]
            file_paths.append((input_path, output_path))

    return file_paths


class Data(object):
    """Class that deals with all the data mess
    """
    def __init__(self, mtrack_list, data_path, input_patch_size):

        self.mtrack_list = mtrack_list
        self.input_patch_size = input_patch_size

        self.data_path = data_path
        
        (self.train_set,
         self.validation_set,
         self.test_set) = self._train_val_test_split()

        self.train_files = get_file_paths(self.train_set, self.data_path)
        self.validation_files = get_file_paths(
            self.validation_set, self.data_path
        )
        self.test_files = get_file_paths(self.test_set, self.data_path)

    def _train_val_test_split(self):
        """Get randomized artist-conditional splits
        """ 
        full_list = []
        for m in self.mtrack_list:
            globbed = get_file_paths([m], self.data_path)
            if len(globbed) > 0:
                full_list.append(m)

        self.full_list = full_list
        mtracks = list(mdb.load_multitracks(full_list))
        test_potentials = [
            m.track_id for m in mtracks if m.dataset_version == 'V1'
        ]
        all_others = [
            m.track_id for m in mtracks if m.dataset_version != 'V1'
        ]

        split1 = utils.artist_conditional_split(
            trackid_list=test_potentials, test_size=0.2,
            num_splits=1, random_state=RANDOM_STATE
        )

        test_set = split1[0]['test']
        remaining_tracks = split1[0]['train'] + all_others

        split2 = utils.artist_conditional_split(
            trackid_list=remaining_tracks, test_size=0.15,
            num_splits=1, random_state=RANDOM_STATE
        )

        train_set = split2[0]['train']
        validation_set = split2[0]['test']

        return train_set, validation_set, test_set

    def get_train_generator(self):
        """return a training data generator
        """
        return keras_generator(
            self.train_files,
            input_patch_size=self.input_patch_size
        )

    def get_validation_generator(self):
        """return a validation data generator
        """
        return keras_generator(
            self.validation_files,
            input_patch_size=self.input_patch_size
        )

    def get_test_generator(self):
        """return a test data generator
        """
        return keras_generator(
            self.test_files,
            input_patch_size=self.input_patch_size
        )

