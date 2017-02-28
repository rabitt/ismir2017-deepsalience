import numpy as np
import medleydb as mdb
from medleydb import download
import librosa
import os


def get_hcqt_params():
    bins_per_octave=120
    n_octaves = 5
    harmonics = [1, 2, 3, 4, 5, 6]
    sr = 22050
    fmin = 32.7
    hop_length = 128
    return bins_per_octave, n_octaves, harmonics, sr, fmin, hop_length


def compute_hcqt(audio_fpath):
    bins_per_octave, n_octaves, harmonics, sr, f_min, hop_length = get_hcqt_params()
    y, fs = librosa.load(audio_fpath, sr=sr)

    cqt_list = []
    shapes = []
    for h in harmonics:
        cqt = librosa.cqt(
            y, sr=fs, hop_length=hop_length, fmin=f_min*float(h),
            n_bins=bins_per_octave*n_octaves,
            bins_per_octave=bins_per_octave
        )
        cqt_list.append(cqt)
        shapes.append(cqt.shape)
    
    shapes_equal = [s == shapes[0] for s in shapes]
    if not all(shapes_equal):
        min_time = np.min([s[1] for s in shapes])
        new_cqt_list = []
        for i, cqt in enumerate(cqt_list):
            new_cqt_list.append(cqt[:, :min_time])
            cqt_list.pop(i)
        cqt_list = new_cqt_list

    log_hcqt = 20.0*np.log10(np.abs(np.array(cqt_list)) + 0.0001)
    log_hcqt = log_hcqt - np.min(log_hcqt)
    log_hcqt = log_hcqt / np.max(log_hcqt)
    return log_hcqt


def get_freq_grid():
    bins_per_octave, n_octaves, harmonics, sr, f_min, hop_length = get_hcqt_params()
    freq_grid = librosa.cqt_frequencies(
        bins_per_octave*n_octaves, f_min, bins_per_octave=bins_per_octave
    )
    return freq_grid


def get_time_grid(n_time_frames):
    bins_per_octave, n_octaves, harmonics, sr, f_min, hop_length = get_hcqt_params()
    time_grid = librosa.core.frames_to_time(
        range(n_time_frames), sr=sr, hop_length=hop_length)
    return time_grid


def grid_to_bins(grid, start_bin_val, end_bin_val):
    bin_centers = (grid[1:] + grid[:-1])/2.0
    bins = np.concatenate([[start_bin_val], bin_centers, [end_bin_val]])
    return bins


def create_annotation_target(freq_grid, time_grid, annotation_times, annotation_freqs):

    time_bins = grid_to_bins(time_grid, 0.0, time_grid[-1])
    freq_bins = grid_to_bins(freq_grid, 0.0, freq_grid[-1])

    annot_time_idx = np.digitize(annotation_times, time_bins) - 1
    annot_freq_idx = np.digitize(annotation_freqs, freq_bins) - 1

    annotation_target = np.zeros((len(freq_grid), len(time_grid)))
    annotation_target[annot_freq_idx, annot_time_idx] = 1

    return annotation_target


def get_all_pitch_annotations(mtrack):
    annot_times = []
    annot_freqs = []
    for stem in mtrack.stems.values():
        data = stem.pitch_annotation
        data2 = stem.pitch_estimate_pyin
        if data is not None:
            annot = data
        elif data2 is not None:
            annot = data2
        else:
            continue

        annot = np.array(annot).T
        annot_times.append(annot[0])
        annot_freqs.append(annot[1])

    annot_times = np.concatenate(annot_times)
    annot_freqs = np.concatenate(annot_freqs)

    return annot_times, annot_freqs


def get_input_output_pairs(mtrack):
    hcqt = compute_hcqt(mtrack.mix_path)

    freq_grid = get_freq_grid()
    time_grid = get_time_grid(len(hcqt[0][0]))

    annot_times, annot_freqs = get_all_pitch_annotations(mtrack)

    annot_target = create_annotation_target(
        freq_grid, time_grid, annot_times, annot_freqs
    )
    plot_annot_target(annot_target, hcqt[0], annot_times, annot_freqs)
    return hcqt, annot_target


def get_input_output_pairs_solo_pitch(audio_path, annot_times, annot_freqs):
    hcqt = compute_hcqt(audio_path)

    freq_grid = get_freq_grid()
    time_grid = get_time_grid(len(hcqt[0][0]))
    annot_target = create_annotation_target(
        freq_grid, time_grid, annot_times, annot_freqs
    )

    return hcqt, annot_target, freq_grid, time_grid


def main(args):
	mtracks = mdb.load_all_multitracks(dataset_version=['V1'])
	for mtrack in mtracks:
		stem = mtrack.predominant_stem
	    if stem is None:
	        continue

	    data = stem.pitch_annotation
	    save_path = os.path.join(
	        args.save_dir,
	        "{}_STEM_{}.npz".format(mtrack.track_id, stem.stem_idx)
	    )

	    if data is not None:
	        print("    > Stem {} {}".format(stem.stem_idx, stem.instrument))
	        annot = np.array(data).T
	    else:
	        continue

	    if os.path.exists(save_path):
	        one_stem_done = True
	        continue

	    if not os.path.exists(stem.audio_path):
	        print("        >downloading stem...")
	        download.download_stem(mtrack, stem.stem_idx)
	        print("         done!")

	    try:
	        data_in, data_out, freq, time = get_input_output_pairs_solo_pitch(
	            stem.audio_path, annot[0], annot[1]
	        )

	        np.savez(save_path, data_in=data_in, data_out=data_out, freq=freq, time=time)
	    except:
	        print("    > Something failed :(")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
        description="Generate feature files for multif0 learning.")
    parser.add_argument("save_dir",
                        type=str,
                        help="Path to save npz files.")

    main(parser.parse_args())
