import glob
import os
import argparse

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default='results')
args = parser.parse_args()

base = args.dir
files = sorted(glob.glob(os.path.join(base, '**', '*.npy'), recursive=True))

if not files:
    print(f'No .npy files found in {base}/')
else:
    for f in files:
        arr = np.load(f)
        txt_path = os.path.splitext(f)[0] + '.txt'
        os.makedirs(os.path.dirname(txt_path), exist_ok=True)

        # Preserve full array shape using array2string
        with open(txt_path, 'w', encoding='utf-8') as out_f:
            out_f.write(np.array2string(arr, separator=', '))

        print(f"Saved {os.path.basename(f)} -> {os.path.basename(txt_path)} (shape={arr.shape}, dtype={arr.dtype})")