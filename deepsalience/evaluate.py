from __future__ import print_function

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import csv
import glob
import librosa
import mir_eval
import numpy as np
import os
import pandas
import scipy

import compute_training_data as C


def test_path():
    """top level path for test data
    """
    return '/scratch/rmb456/multif0_ismir2017/test_data/'


def save_multif0_output(times, freqs, output_path):
    """save multif0 output to a csv file
    """
    with open(output_path, 'w') as fhandle:
        csv_writer = csv.writer(fhandle, delimiter='\t')
        for t, f in zip(times, freqs):
            row = [t]
            row.extend(f)
            csv_writer.writerow(row)


def get_best_thresh(dat, model):
    """Use validation set to get the best threshold value
    """

    # get files for this test set
    validation_files = dat.validation_files
    test_set_path = test_path()

    thresh_vals = np.arange(0.1, 1.0, 0.1)
    thresh_scores = {t: [] for t in thresh_vals}
    for npy_file, _ in validation_files:

        file_keys = os.path.basename(npy_file).split('_')[:2]
        label_file = glob.glob(
            os.path.join(
                test_set_path, 'mdb_test',
                "{}*{}.txt".format(file_keys[0], file_keys[1]))
        )[0]

        # generate prediction on numpy file
        predicted_output, input_hcqt = \
            get_single_test_prediction(npy_file, model)

        # load ground truth labels
        ref_times, ref_freqs = \
            mir_eval.io.load_ragged_time_series(label_file)

        for thresh in thresh_vals:
            # get multif0 output from prediction
            est_times, est_freqs = \
                pitch_activations_to_mf0(predicted_output, thresh)

            # get multif0 metrics and append
            scores = mir_eval.multipitch.evaluate(
                ref_times, ref_freqs, est_times, est_freqs)
            thresh_scores[thresh].append(scores['Accuracy'])

    avg_thresh = [np.mean(thresh_scores[t]) for t in thresh_vals]
    best_thresh = thresh_vals[np.argmax(avg_thresh)]
    print("Best Threshold is {}".format(best_thresh))
    print("Best validation accuracy is {}".format(np.max(avg_thresh)))
    print("Validation accuracy at 0.5 is {}".format(np.mean(thresh_scores[0.5])))

    return best_thresh


def score_on_test_set(test_set_name, model, save_path, thresh=0.5):
    """score a model on all files in a named test set
    """

    # get files for this test set
    test_set_path = os.path.join(test_path(), test_set_name)
    test_npy_files = glob.glob(os.path.join(test_set_path, '*.npy'))

    all_scores = []
    for npy_file in sorted(test_npy_files):
        # get input npy file and ground truth label pair
        file_keys = os.path.basename(npy_file).split('.')[0]
        label_file = glob.glob(
            os.path.join(test_set_path, "{}.txt".format(file_keys))
        )[0]

        # generate prediction on numpy file
        predicted_output, input_hcqt = \
            get_single_test_prediction(npy_file, model)

        # save plot for first example
        if len(all_scores) == 0:
            plot_save_path = os.path.join(
                save_path,
                "{}_{}_plot_output.pdf".format(file_keys[0], file_keys[1])
            )

            plt.figure(figsize=(15, 15))

            plt.subplot(2, 1, 1)
            plt.imshow(input_hcqt[0, :, :, 1], origin='lower')
            plt.axis('auto')
            plt.colorbar()

            plt.subplot(2, 1, 2)
            plt.imshow(predicted_output, origin='lower')
            plt.axis('auto')
            plt.colorbar()

            plt.savefig(plot_save_path, format='pdf')
            plt.close()

        # save prediction
        np.save(
            os.path.join(
                save_path,
                "{}_{}_prediction.npy".format(file_keys[0], file_keys[1])
            ),
            predicted_output.astype(np.float32)
        )

        # get multif0 output from prediction
        est_times, est_freqs = pitch_activations_to_mf0(
            predicted_output, thresh
        )

        # save multif0 output
        save_multif0_output(
            est_times, est_freqs,
            os.path.join(
                save_path,
                "{}_{}_prediction.txt".format(file_keys[0], file_keys[1])
            )
        )

        # load ground truth labels
        ref_times, ref_freqs = \
            mir_eval.io.load_ragged_time_series(label_file)

        # get multif0 metrics and append
        scores = mir_eval.multipitch.evaluate(
            ref_times, ref_freqs, est_times, est_freqs)
        scores['track'] = '_'.join(file_keys)
        all_scores.append(scores)

    # save scores to data frame
    scores_path = os.path.join(
        save_path, '{}_all_scores.csv'.format(test_set_name)
    )
    score_summary_path = os.path.join(
        save_path, "{}_score_summary.csv".format(test_set_name)
    )
    df = pandas.DataFrame(all_scores)
    df.to_csv(scores_path)
    df.describe().to_csv(score_summary_path)
    print(df.describe())


def get_model_metrics(data_object, model, model_scores_path):
    """Get model loss and metrics on train, validation and test generators
    """
    train_generator = data_object.get_train_generator()
    validation_generator = data_object.get_validation_generator()
    test_generator = data_object.get_test_generator()

    train_eval = model.evaluate_generator(
        train_generator, 1000, max_q_size=10
    )
    valid_eval = model.evaluate_generator(
        validation_generator, 1000, max_q_size=10
    )
    test_eval = model.evaluate_generator(
        test_generator, 1000, max_q_size=10
    )

    df = pandas.DataFrame(
        [train_eval, valid_eval, test_eval],
        index=['train', 'validation', 'test']
    )
    print(df)
    df.to_csv(model_scores_path)


def get_paths(save_dir, save_key):
    """For a given save dir and save key define lots of save paths
    """
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
    """create and save plot of loss and metrics across epochs
    """
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


def pitch_activations_to_mf0(pitch_activation_mat, thresh):
    """Convert a pitch activation map to multif0 by thresholding peak values
    at thresh
    """
    freqs = C.get_freq_grid()
    times = C.get_time_grid(pitch_activation_mat.shape[1])

    peak_thresh_mat = np.zeros(pitch_activation_mat.shape)
    peaks = scipy.signal.argrelmax(pitch_activation_mat, axis=0)
    peak_thresh_mat[peaks] = pitch_activation_mat[peaks]

    idx = np.where(peak_thresh_mat >= thresh)

    est_freqs = [[] for _ in range(len(times))]
    for f, t in zip(idx[0], idx[1]):
        est_freqs[t].append(freqs[f])

    est_freqs = [np.array(lst) for lst in est_freqs]
    return times, est_freqs


# def compute_metrics(predicted_mat, true_mat):
#     """Score two pitch activation maps (predictions against ground truth)
#     """
#     ref_times, ref_freqs = pitch_activations_to_mf0(true_mat, 1)
#     est_times, est_freqs = pitch_activations_to_mf0(predicted_mat, 0.5)

#     scores = mir_eval.multipitch.evaluate(
#         ref_times, ref_freqs, est_times, est_freqs
#     )
#     return scores


def get_single_test_prediction(model, npy_file=None, audio_file=None):
    """Generate output from a model given an input numpy file
    """
    if npy_file is not None:
        input_hcqt = np.load(npy_file)
    elif audio_file is not None:
        input_hcqt = (C.compute_hcqt(audio_file)).astype(np.float32)
    else:
        raise ValueError("one of npy_file or audio_file must be specified")

    input_hcqt = input_hcqt.transpose(1, 2, 0)[np.newaxis, :, :, :]

    n_t = input_hcqt.shape[2]
    t_slices = list(np.arange(0, n_t, 5000))
    output_list = []
    for t in t_slices:
        output_list.append(
            model.predict(input_hcqt[:, :, t:t+5000, :])[0, :, :]
        )

    predicted_output = np.hstack(output_list)
    return predicted_output, input_hcqt


def plot_prediction(input_hcqt, predicted_output, true_output, save_path):
    """Plot a trip of input, prediction and ground truth and save it
    """
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
