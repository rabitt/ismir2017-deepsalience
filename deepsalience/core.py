"""Core data and model architecture classes
"""
import glob
import keras
import medleydb as mdb
from medleydb import utils
import numpy as np
import os
import pescador


RANDOM_STATE = 42


def keras_generator(data_list, patch_size=(20, 20), with_replacement=True,
                    batch_size=1024):
    streams = []
    for fpath in data_list:
        streams.append(
            pescador.Streamer(
                patch_generator, fpath, patch_size=patch_size
            )
        )

    stream_mux = pescador.Mux(
        streams, 2, with_replacement=with_replacement, lam=None
    )

    batch_generator = pescador.buffer_batch(stream_mux.generate(), batch_size)

    for batch in batch_generator:
        yield (batch['X'], batch['Y'])


def __grab_patch_output(f, t, n_f, n_t, y_data):
    return y_data[f: f + n_f, t: t + n_t].reshape(1, n_f, n_t)


def __grab_patch_input(f, t, n_f, n_t, n_harms, x_data):
    return np.transpose(
        x_data[:, f: f + n_f, t: t + n_t], (1, 2, 0)
    ).reshape(1, n_f, n_t, n_harms)


def patch_generator(fpath, patch_size):
    data = np.load(fpath, mmap_mode='r')
    n_harms, n_freqs, n_times = data['data_in'].shape
    n_f, n_t = patch_size

    while True:
        f = np.random.randint(0, n_freqs - n_f)
        t = np.random.randint(0, n_times - n_t)
        x = __grab_patch_input(f, t, n_f, n_t, n_harms, data['data_in'])
        y = __grab_patch_output(f, t, n_f, n_t, data['data_out'])
        yield dict(X=x, Y=y)


def stride_tf(fpath, patch_size):
    data = np.load(fpath, mmap_mode='r')
    n_harms, n_freqs, n_times = data['data_in'].shape
    n_f, n_t = patch_size

    f_indices = np.arange(0, n_freqs - n_f, n_f)
    t_indices = np.arange(0, n_times - n_t, n_t)

    for f in f_indices:
        for t in t_indices:
            x = __grab_patch_input(f, t, n_f, n_t, n_harms, data['data_in'])
            y = __grab_patch_output(f, t, n_f, n_t, data['data_out'])
            yield dict(X=x, Y=y, t=t, f=f)


def get_file_paths(mtrack_list, data_path):
    file_paths = []
    for track_id in mtrack_list:
        file_paths.extend(
            glob.glob(os.path.join(data_path, "{}*.npz".format(track_id)))
        )
    return file_paths


class Data(object):

    def __init__(self, mtrack_list, data_path, patch_size=(20, 20),
                 batch_size=1024):

        self.mtrack_list = mtrack_list
        self.patch_size = patch_size
        self.batch_size = batch_size

        (self.train_set,
         self.validation_set,
         self.test_set) = self._train_val_test_split()

        self.data_path = data_path

        self.train_files = get_file_paths(self.train_set, self.data_path)
        self.validation_files = get_file_paths(
            self.validation_set, self.data_path
        )
        self.test_files = get_file_paths(self.test_set, self.data_path)

    def _train_val_test_split(self):
        mtracks = mdb.load_multitracks(self.mtrack_list)
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
        return keras_generator(
            self.train_files, patch_size=self.patch_size,
            batch_size=self.batch_size
        )

    def get_validation_generator(self):
        return keras_generator(
            self.validation_files, patch_size=self.patch_size,
            batch_size=self.batch_size
        )

    def get_test_generator(self):
        return keras_generator(
            self.test_files, patch_size=self.patch_size,
            batch_size=self.batch_size
        )


class Model(object):

    def __init__(self, loss, input_shape, optimizer='sgd',
                 samples_per_epoch=102400, n_epochs=10, n_val_samples=1024):
        self.model = self._build_model()
        self.loss = loss
        self.input_shape = input_shape
        self.optimizer = optimizer
        self.samples_per_epoch = samples_per_epoch
        self.n_epochs = n_epochs
        self.n_val_samples = n_val_samples

    def model_definition(self):
        raise NotImplementedError

    def _build_model(self):
        model = self.model_definition()
        model.compile(loss=self.loss, optimizer=self.optimizer)
        return model

    def fit(self, train_generator, validation_generator):
        history = self.model.fit_generator(
            train_generator, self.samples_per_epoch, self.n_epochs, verbose=1,
            validation_data=validation_generator,
            n_val_samples=self.n_val_samples
        )
        return history

    def predict(self, test_data):
        n_harms, n_freqs, n_times = data_in.shape
        n_f, n_t = (20, 20)

        prediction = np.zeros((n_freqs, n_times))

        cqt_patch_generator = stride_cqt(data_in)

        for d in cqt_patch_generator:
            f = d['f']
            t = d['t']
            y_pred = model.predict(d['X'].reshape(1, n_f, n_t, n_harms)).reshape(n_f, n_t)
            prediction[f: f + n_f, t: t + n_t] = y_pred

        return prediction

    @property
    def id(self):
        pass
