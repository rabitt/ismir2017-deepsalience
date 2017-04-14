"""Core data and model architecture classes
"""
from __future__ import print_function

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
np.random.seed(1337)
from keras import backend as K
import pescador
import keras
import glob
import medleydb as mdb
from medleydb import utils
import os
import pandas


import mir_eval
import compute_training_data as C

from tensorflow.python.client import device_lib
device_lib.list_local_devices()

RANDOM_STATE = 42


def get_model_metrics(data_object, model, model_scores_path):
    train_generator = data_object.get_train_generator()
    validation_generator = data_object.get_validation_generator()
    test_generator = data_object.get_test_generator()

    train_eval = model.evaluate_generator(train_generator, 5000, max_q_size=10)
    valid_eval = model.evaluate_generator(validation_generator, 5000, max_q_size=10)
    test_eval = model.evaluate_generator(test_generator, 5000, max_q_size=10)

    df = pandas.DataFrame([train_eval, valid_eval, test_eval], index=['train', 'validation', 'test'])
    print(df)
    df.to_csv(model_scores_path)


def get_all_multif0_metrics(test_files, model, save_dir, scores_path, score_summary_path):
    all_scores = []
    for test_pair in test_files:
        pair_key = os.path.basename(test_pair[0])
        print("    > {}".format(pair_key))
        save_path = os.path.join(
            save_dir, "{}.pdf".format(os.path.basename(test_pair[0]).split('.')[0])
        )
        predicted_output, true_output = generate_prediction(
            test_pair, model, save_path=save_path
        )

        scores = compute_metrics(predicted_output, true_output)
        scores['track'] = pair_key
        all_scores.append(scores)

    df = pandas.DataFrame(all_scores)
    df.to_csv(scores_path)
    df.describe().to_csv(score_summary_path)
    print(df.describe())


def get_paths(save_dir, save_key):
    save_path = os.path.join(save_dir, save_key)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    model_save_path = os.path.join(save_path, "{}.pkl".format(save_key))
    plot_save_path = os.path.join(save_path, "{}_loss.pdf".format(save_key))
    model_scores_path = os.path.join(
        save_path, "{}_model_scores.csv".format(save_key))
    scores_path = os.path.join(save_path, "{}_scores.csv".format(save_key))
    score_summary_path = os.path.join(
        save_path, "{}_score_summary.csv".format(save_key))
    return (save_path, model_save_path, plot_save_path,
            model_scores_path, scores_path, score_summary_path)


def plot_metrics_epochs(history, plot_save_path):
    plt.figure(figsize=(15, 15))

    plt.subplot(3, 1, 1)
    plt.plot(history.history['mean_squared_error'])
    plt.plot(history.history['val_mean_squared_error'])
    plt.title('mean squared error')
    plt.ylabel('mean squared error')
    plt.xlabel('epoch')
    plt.legend(['train', 'validate'], loc='upper left')

    plt.subplot(3, 1, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validate'], loc='upper left')

    plt.subplot(3, 1, 3)
    plt.plot(history.history['soft_binary_accuracy'])
    plt.plot(history.history['val_soft_binary_accuracy'])
    plt.title('soft_binary_accuracy')
    plt.ylabel('soft_binary_accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validate'], loc='upper left')

    plt.savefig(plot_save_path, format='pdf')
    plt.close()


def keras_loss():
    return bkld


def keras_metrics():
    return ['mse', soft_binary_accuracy]


def compute_metrics(predicted_mat, true_mat):
    freqs = C.get_freq_grid()
    times = C.get_time_grid(predicted_mat.shape[1])
    ref_idx = np.where(true_mat == 1)
    est_idx = np.where(predicted_mat > 0.5)

    est_freqs = [[] for _ in range(len(times))]
    for f, t in zip(est_idx[0], est_idx[1]):
        est_freqs[t].append(freqs[f])

    ref_freqs = [[] for _ in range(len(times))]
    for f, t in zip(ref_idx[0], ref_idx[1]):
        ref_freqs[t].append(freqs[f])

    est_freqs = [np.array(lst) for lst in est_freqs]
    ref_freqs = [np.array(lst) for lst in ref_freqs]

    scores = mir_eval.multipitch.evaluate(times, ref_freqs, times, est_freqs)
    return scores


def generate_prediction(test_pair, model, save_path=None):
    true_output = np.load(test_pair[1])
    input_hcqt = np.load(test_pair[0]).transpose(1, 2, 0)[np.newaxis, :, :, :]

    n_t = input_hcqt.shape[2]
    t_slices = list(np.arange(0, n_t, 5000))
    output_list = []
    for t in t_slices:
        output_list.append(
            model.predict(input_hcqt[:, :, t:t+5000, :])[0, :, :]
        )

    predicted_output = np.hstack(output_list)

    if save_path is not None:
        plot_prediction(input_hcqt, predicted_output, true_output, save_path)

    return predicted_output, true_output


def plot_prediction(input_hcqt, predicted_output, true_output, save_path):
    plt.figure(figsize=(15, 15))

    plt.subplot(3, 1, 1)
    plt.imshow(input_hcqt[0, :, :, 1], origin='lower')
    plt.axis('auto')
    plt.colorbar()

    plt.subplot(3, 1, 2)
    plt.imshow(predicted_output, origin='lower')
    plt.axis('auto')
    plt.colorbar()

    plt.subplot(3, 1, 3)
    plt.imshow(true_output, origin='lower')
    plt.axis('auto')
    plt.colorbar()

    plt.savefig(save_path, format='pdf')
    plt.close()


def bkld(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), 1.0 - K.epsilon())
    y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
    return K.mean(K.mean(
        -1.0*y_true* K.log(y_pred) - (1.0 - y_true) * K.log(1.0 - y_pred),
        axis=-1), axis=-1)


def soft_binary_accuracy(y_true, y_pred):
    return K.mean(K.mean(
        K.equal(K.round(y_true), K.round(y_pred)), axis=-1), axis=-1)


def keras_generator(data_list, input_patch_size, output_patch_size,
                    with_replacement=True,
                    batch_size=32):

    streams = []
    for fpath_in, fpath_out in data_list:
        streams.append(
            pescador.Streamer(
                patch_generator, fpath_in, fpath_out,
                input_patch_size=input_patch_size,
                output_patch_size=output_patch_size,
                batch_size=batch_size
            )
        )

    stream_mux = pescador.Mux(
        streams, 10, with_replacement=with_replacement, lam=500
    )

    for batch in stream_mux.tuples('X', 'Y'):
        yield batch


def __grab_patch_output(f, t, n_f, n_t, y_data):
    return y_data[f: f + n_f, t: t + n_t][np.newaxis, :, :]


def __grab_patch_input(f, t, n_f, n_t, n_harms, x_data):
    return np.transpose(
        x_data[:, f: f + n_f, t: t + n_t], (1, 2, 0)
    )[np.newaxis, :, :, :]


def patch_generator(fpath_in, fpath_out, input_patch_size, output_patch_size, batch_size):
    data_in = np.load(fpath_in)
    data_out = np.load(fpath_out)

    n_harms, n_freqs, n_times = data_in.shape
    n_f_in, n_t_in = input_patch_size
    n_f_out, n_t_out = output_patch_size

    f_shift = n_f_in - n_f_out
    t_shift = n_t_in - n_t_out

    while True:
        # f = 0 if n_f_in == n_freqs else np.random.randint(0, n_freqs - n_f_in)
        f = 0
        t = np.random.randint(0, n_times - n_t_in)

        x = __grab_patch_input(
            f, t, n_f_in, n_t_in, n_harms, data_in
        )
        y = __grab_patch_output(
            f + f_shift, t + t_shift, n_f_out, n_t_out, data_out
        )
        yield dict(X=x, Y=y)


def get_file_paths(mtrack_list, data_path):
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

    def __init__(self, mtrack_list, data_path, input_patch_size,
                 output_patch_size, batch_size):

        self.mtrack_list = mtrack_list
        self.input_patch_size = input_patch_size
        self.output_patch_size = output_patch_size
        self.batch_size = batch_size

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
        mtracks = mdb.load_multitracks(self.mtrack_list)

        test_potentials = []
        test_only = []
        all_others = []
        full_list = []

        for mtrack in mtracks:
            globbed = get_file_paths([mtrack.track_id], self.data_path)
            if len(globbed) == 0:
                continue

            full_list.append(mtrack.track_id)

            if mtrack.dataset_version == 'V1':
                test_potentials.append(mtrack.track_id)
            elif mtrack.dataset_version == 'BACH10':
                test_only.append(mtrack.track_id)
            else:
                all_others.append(mtrack.track_id)

        split1 = utils.artist_conditional_split(
            trackid_list=test_potentials, test_size=0.2,
            num_splits=1, random_state=RANDOM_STATE
        )

        test_set = split1[0]['test']
        test_set.extend(test_only)
        remaining_tracks = split1[0]['train'] + all_others

        split2 = utils.artist_conditional_split(
            trackid_list=remaining_tracks, test_size=0.15,
            num_splits=1, random_state=RANDOM_STATE
        )

        train_set = split2[0]['train']
        validation_set = split2[0]['test']

        return train_set, validation_set, test_set

    def get_train_generator(self):
        return keras_generator(
            self.train_files,
            input_patch_size=self.input_patch_size,
            output_patch_size=self.output_patch_size,
            batch_size=self.batch_size
        )

    def get_validation_generator(self):
        return keras_generator(
            self.validation_files,
            input_patch_size=self.input_patch_size,
            output_patch_size=self.output_patch_size,
            batch_size=self.batch_size
        )

    def get_test_generator(self):
        return keras_generator(
            self.test_files,
            input_patch_size=self.input_patch_size,
            output_patch_size=self.output_patch_size,
            batch_size=self.batch_size
        )

