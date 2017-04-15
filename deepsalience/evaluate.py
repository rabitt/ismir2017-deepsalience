from __future__ import print_function

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import librosa
import mir_eval
import numpy as np
import os
import pandas

import compute_training_data as C


def get_model_metrics(data_object, model, model_scores_path):
    train_generator = data_object.get_train_generator()
    validation_generator = data_object.get_validation_generator()
    test_generator = data_object.get_test_generator()

    train_eval = model.evaluate_generator(
        train_generator, 5000, max_q_size=10
    )
    valid_eval = model.evaluate_generator(
        validation_generator, 5000, max_q_size=10
    )
    test_eval = model.evaluate_generator(
        test_generator, 5000, max_q_size=10
    )

    df = pandas.DataFrame(
        [train_eval, valid_eval, test_eval],
        index=['train', 'validation', 'test']
    )
    print(df)
    df.to_csv(model_scores_path)


def get_all_multif0_metrics(test_files, model, save_dir, scores_path,
                            score_summary_path, create_pred=False):
    all_scores = []
    for test_pair in test_files:
        pair_key = os.path.basename(test_pair[0])
        print("    > {}".format(pair_key))
        if create_pred:
            save_path = os.path.join(
                save_dir, "{}.pdf".format(
                    os.path.basename(test_pair[0]).split('.')[0]
                )
            )
        else:
            save_path = None
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
        if not os.path.exists(save_path):
            plot_prediction(
                input_hcqt, predicted_output, true_output, save_path
            )

        freqs = C.get_freq_grid()
        times = C.get_time_grid(predicted_output.shape[1])
        fs = 16000
        try:
            y_synth = mir_eval.sonify.time_frequency(
                predicted_output, freqs, times, fs
            )
            librosa.output.write_wav(
                "{}.wav".format(save_path), y_synth, fs, norm=True
            )
        except:
            print("    > unable to synthesize {}".format(save_path))

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
