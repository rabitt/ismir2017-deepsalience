"""Script to predict deepsalience output from audio"""
from __future__ import print_function

import argparse
import librosa
import numpy as np
import os
import scipy
import csv

from keras.models import Model
from keras.layers import Dense, Input, Reshape, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.models import load_model

TASKS = ['bass', 'melody1', 'melody2', 'melody3', 'multif0', 'pitch', 'vocal']
BINS_PER_OCTAVE = 60
N_OCTAVES = 6
HARMONICS = [0.5, 1, 2, 3, 4, 5]
SR = 22050
FMIN = 32.7
HOP_LENGTH = 256


def compute_hcqt(audio_fpath):
    """Compute the harmonic CQT from a given audio file

    Parameters
    ----------
    audio_fpath : str
        path to audio file

    Returns
    -------
    hcqt : np.ndarray
        Harmonic cqt
    time_grid : np.ndarray
        List of time stamps in seconds
    freq_grid : np.ndarray
        List of frequency values in Hz

    """
    y, fs = librosa.load(audio_fpath, sr=SR)

    cqt_list = []
    shapes = []
    for h in HARMONICS:
        cqt = librosa.cqt(
            y, sr=fs, hop_length=HOP_LENGTH, fmin=FMIN*float(h),
            n_bins=BINS_PER_OCTAVE*N_OCTAVES,
            bins_per_octave=BINS_PER_OCTAVE
        )
        cqt_list.append(cqt)
        shapes.append(cqt.shape)

    shapes_equal = [s == shapes[0] for s in shapes]
    if not all(shapes_equal):
        min_time = np.min([s[1] for s in shapes])
        new_cqt_list = []
        for i in range(len(cqt_list)):
            new_cqt_list.append(cqt_list[i][:, :min_time])
        cqt_list = new_cqt_list

    log_hcqt = ((1.0/80.0) * librosa.core.amplitude_to_db(
        np.abs(np.array(cqt_list)), ref=np.max)) + 1.0

    freq_grid = librosa.cqt_frequencies(
        BINS_PER_OCTAVE*N_OCTAVES, FMIN, bins_per_octave=BINS_PER_OCTAVE
    )

    time_grid = librosa.core.frames_to_time(
        range(log_hcqt.shape[2]), sr=SR, hop_length=HOP_LENGTH
    )

    return log_hcqt, freq_grid, time_grid


def bkld(y_true, y_pred):
    """KL Divergence where both y_true an y_pred are probabilities
    """
    y_true = K.clip(y_true, K.epsilon(), 1.0 - K.epsilon())
    y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
    return K.mean(K.mean(
        -1.0*y_true* K.log(y_pred) - (1.0 - y_true) * K.log(1.0 - y_pred),
        axis=-1), axis=-1)


def model_def():
    """Created compiled Keras model

    Returns
    -------
    model : Model
        Compiled keras model
    """
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
    y5 = Conv2D(8, (70, 3), padding='same', activation='relu', name='distribute')(y4a)
    y5a = BatchNormalization()(y5)
    y6 = Conv2D(1, (1, 1), padding='same', activation='sigmoid', name='squishy')(y5a)
    predictions = Lambda(lambda x: K.squeeze(x, axis=3))(y6)

    model = Model(inputs=inputs, outputs=predictions)
    model.compile(loss=bkld, metrics=['mse'], optimizer='adam')
    return model


def load_model(task):
    """Load a precompiled, pretrained model

    Parameters
    ----------
    task : str
        One of
            -'bass'
            -'melody1'
            -'melody2'
            -'melody3'
            -'multif0'
            -'pitch'
            -'vocal'

    Returns
    -------
    model : Model
        Pretrained, precompiled Keras model

    """
    model = model_def()
    if task not in TASKS:
        raise ValueError("task must be one of {}".format(TASKS))

    weights_path = os.path.join('weights', '{}.h5'.format(task))
    if not os.path.exists(weights_path):
        raise IOError(
            "Cannot find weights path {} for this task.".format(weights_path))

    model.load_weights(weights_path)
    return model


def get_single_test_prediction(model, input_hcqt):
    """Generate output from a model given an input numpy file

    Parameters
    ----------
    model : Model
        Pretrained model
    input_hcqt : np.ndarray
        HCQT

    Returns
    -------
    predicted_output : np.ndarray
        Matrix of predictions

    """
    input_hcqt = input_hcqt.transpose(1, 2, 0)[np.newaxis, :, :, :]

    n_t = input_hcqt.shape[2]
    n_slices = 2000
    t_slices = list(np.arange(0, n_t, n_slices))
    output_list = []
    for i, t in enumerate(t_slices):
        print("   > {} / {}".format(i + 1, len(t_slices)))
        prediction = model.predict(input_hcqt[:, :, t:t+n_slices, :])
        output_list.append(prediction[0, :, :])

    predicted_output = np.hstack(output_list)

    return predicted_output


def get_multif0(pitch_activation_mat, freq_grid, time_grid, thresh=0.3):
    """Compute multif0 output containing all peaks in the output that
       fall above thresh

    Parameters
    ----------
    pitch_activation_mat : np.ndarray
        Deep salience prediction
    freq_grid : np.ndarray
        Frequency values
    time_grid : np.ndarray
        Time values
    thresh : float, default=0.3
        Likelihood threshold

    Returns
    -------
    times : np.ndarray
        Time values
    freqs : list
        List of lists of frequency values

    """
    peak_thresh_mat = np.zeros(pitch_activation_mat.shape)
    peaks = scipy.signal.argrelmax(pitch_activation_mat, axis=0)
    peak_thresh_mat[peaks] = pitch_activation_mat[peaks]

    idx = np.where(peak_thresh_mat >= thresh)

    est_freqs = [[] for _ in range(len(time_grid))]
    for f, t in zip(idx[0], idx[1]):
        est_freqs[t].append(freq_grid[f])

    est_freqs = [np.array(lst) for lst in est_freqs]
    return time_grid, est_freqs


def get_singlef0(pitch_activation_mat, freq_grid, time_grid, thresh=0.3,
                 use_neg=True):
    """Compute single-f0 output containing the maximum likelihood per time frame.
       Frames with no likelihoods above the threshold are given negative values.

    Parameters
    ----------
    pitch_activation_mat : np.ndarray
        Deep salience prediction
    freq_grid : np.ndarray
        Frequency values
    time_grid : np.ndarray
        Time values
    thresh : float, default=0.3
        Likelihood threshold
    use_neg : bool
        If True, frames with no value above the threshold the frequency
        are given negative values of the frequency with the largest liklihood.
        If False, those frames are given the value 0.0

    Returns
    -------
    times : np.ndarray
        Time values
    freqs : np.ndarray
        Frequency values

    """
    max_idx = np.argmax(pitch_activation_mat, axis=0)
    est_freqs = []
    for i, f in enumerate(max_idx):
        if pitch_activation_mat[f, i] < thresh:
            if use_neg:
                est_freqs.append(-1.0*freq_grid[f])
            else:
                est_freqs.append(0.0)
        else:
            est_freqs.append(freq_grid[f])
    est_freqs = np.array(est_freqs)
    return time_grid, est_freqs


def save_multif0_output(times, freqs, output_path):
    """Save multif0 output to a csv file

    Parameters
    ----------
    times : np.ndarray
        array of time values
    freqs : list of lists
        list of lists of frequency values
    output_path : str
        path to save output

    """
    with open(output_path, 'w') as fhandle:
        csv_writer = csv.writer(fhandle, delimiter='\t')
        for t, f in zip(times, freqs):
            row = [t]
            row.extend(f)
            csv_writer.writerow(row)


def save_singlef0_output(times, freqs, output_path):
    """Save singlef0 output to a csv file

    Parameters
    ----------
    times : np.ndarray
        array of time values
    freqs : np.ndarray
        array of frequency values
    output_path : str
        path to save output

    """
    with open(output_path, 'w') as fhandle:
        csv_writer = csv.writer(fhandle, delimiter='\t')
        for t, f in zip(times, freqs):
            csv_writer.writerow([t, f])


def compute_output(hcqt, time_grid, freq_grid, task, output_format, threshold,
                   use_neg, save_dir, save_name):
    """Comput output for a given task

    Parameters
    ----------
    hcqt : np.ndarray
        harmonic cqt
    time_grid : np.ndarray
        array of times
    freq_grid : np.ndarray
        array of frequencies
    task : str
        which task to compute
    output_format : str
        specify whehter to save output as singlef0, multif0 or salience
    threshold : float
        amplitude threshold for multif0 and singlef0 output
    use_neg : bool
        whether to report negative frequency values in singlef0 output
    save_dir : str
        Path to folder to save output
    save_name : str
        Output file basename

    """
    model = load_model(task)

    print("Computing salience...")
    pitch_activation_mat = get_single_test_prediction(model, hcqt)

    print("Saving output...")
    if output_format == 'singlef0':
        times, freqs = get_singlef0(
            pitch_activation_mat, freq_grid, time_grid, thresh=threshold,
            use_neg=use_neg
        )
        save_path = os.path.join(
            save_dir, "{}_{}_singlef0.csv".format(save_name, task))
        save_singlef0_output(times, freqs, save_path)
    elif output_format == 'multif0':
        times, freqs = get_multif0(
            pitch_activation_mat, freq_grid, time_grid, thresh=threshold)
        save_path = os.path.join(
            save_dir, "{}_{}_multif0.csv".format(save_name, task))
        save_multif0_output(times, freqs, save_path)
    else:
        save_path = os.path.join(
            save_dir, "{}_{}_salience.npz".format(save_name, task))
        np.savez(save_path, salience=pitch_activation_mat, times=time_grid,
            freqs=freq_grid)

    print("Done!")


def load_model_melody1():
    """Load the melody1 model and return it (used by Replicate)

    Returns
    -------
    model : Model
        The pretrained melody1 model
    """
    return load_model("melody1")


def infer_example_melody1(model, audio_path):
    """Run a single inference of the melody1 model on an audio file

    Parameters
    ----------
    model : Model
        The pretrained melody1 model
    audio_path : str
        Path to audio file to extract melody from

    Returns
    -------
    (times, freqs) : Tuple[np.ndarray, np.ndarray]
        Time grid and predicted frequencies
    """
    hcqt, freq_grid, time_grid = compute_hcqt(audio_path)
    pitch_activation_mat = get_single_test_prediction(model, hcqt)
    times, freqs = get_singlef0(
        pitch_activation_mat, freq_grid, time_grid, thresh=0.3,
        use_neg=True
    )
    return times, freqs


def main(args):
    if args.task not in ['all'] + TASKS:
        raise ValueError("task must be 'all' or one of {}".format(TASKS))

    save_name = os.path.basename(args.audio_fpath).split('.')[0]

    # this is slow for long audio files
    print("Computing HCQT...")
    hcqt, freq_grid, time_grid = compute_hcqt(args.audio_fpath)


    if args.task == 'all':
        for task in TASKS:
            print("[Computing {} output]".format(task))
            compute_output(
                hcqt, time_grid, freq_grid, task, args.output_format,
                args.threshold, args.use_neg, args.save_dir, save_name)
    else:
        compute_output(
            hcqt, time_grid, freq_grid, args.task, args.output_format,
            args.threshold, args.use_neg, args.save_dir, save_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict deep salience output for a given task")
    parser.add_argument("audio_fpath",
                        type=str,
                        help="Path to input audio file.")
    parser.add_argument("task",
                        type=str,
                        help="Task to compute one of "
                        "all, bass, melody1, melody2, melody3, "
                        "multif0, pitch, vocal.")
    parser.add_argument("save_dir",
                        type=str,
                        help="Path to folder for saving output")
    parser.add_argument("-f", "--output_format",
                        type=str,
                        choices=['singlef0', 'multif0', 'salience'],
                        default='salience',
                        help="Which format to save output. "
                        "singlef0 saves a csv of single f0 values. "
                        "mulif0 saves a csv of multif0 values. "
                        "salience (default) saves a npz file of the "
                        "salience matrix.")
    parser.add_argument("-t", "--threshold",
                        type=float,
                        default=0.3,
                        help="Amplitude threshold. Only used when "
                        "output_format is singlef0 or multif0")
    parser.add_argument("-n", "--use_neg",
                        type=bool,
                        default=True,
                        help="If True, report unvoiced frames with negative values. "
                        "This is only used when output_format is singlef0.")

    main(parser.parse_args())
