import argparse
import csv
import numpy as np

import compute_training_data as C


def main(args):
    activation_map = np.load(args.npy_fpath)
    freqs = C.get_freq_grid()
    times = C.get_time_grid(activation_map.shape[1])
    idx = np.where(activation_map == 1)

    ref_freqs = [[] for _ in range(len(times))]
    for f, t in zip(idx[0], idx[1]):
        ref_freqs[t].append(freqs[f])

    with open(args.output_fpath, 'w') as fhandle:
        csv_writer = csv.writer(fhandle, delimiter='\t')
        for t, f in zip(times, ref_freqs):
            csv_writer.writerow([t] + f) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert npy output files to multif0 mirex format")
    parser.add_argument("npy_fpath",
                        type=str,
                        help="Path to npy file.")
    parser.add_argument("output_fpath",
                        type=str,
                        help="Path to save output")
    main(parser.parse_args())
