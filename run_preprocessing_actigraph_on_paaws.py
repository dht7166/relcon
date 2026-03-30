"""
This code go over the entire released PAAWS dataset to grab data for pre-training.
Use all sensors, except for the Wrist and Thigh sensor - which will be used for downstream task.
"""

import sys, os, glob, subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

PYTHON     = sys.executable   # uses whichever env launched this script (.env/bin/python)
SCRIPT     = 'relcon/data/process/process_actigraph_pretrain.py'
OUTPUT_DIR = 'relcon/data/datasets/paaws_motifdist'
N_WORKERS  = 4   # tune to available cores / I/O bandwidth

path_to_file   = os.path.join(sys.argv[1], 'DS_*', 'accel', '*.csv') # provide full path to sys.argv[1] e.g., /path/PAAWS_FreeLiving/
list_actigraph = glob.glob(path_to_file)
list_actigraph = [x for x in list_actigraph if 'Thigh' not in x and 'Wrist' not in x]
use_for_val    = lambda x: x > 0.7 * len(list_actigraph)


def process_one(index, actigraph):
    filename  = os.path.basename(actigraph)
    ID, condition, sensor = filename[:-4].split('-')  # DS_ID-[Lab/Free]-[Sensor].csv
    unique_id = f'{ID}-{condition}-{sensor}'
    tag       = '--val' if use_for_val(index) else '--train'
    result    = subprocess.run(
        [PYTHON, SCRIPT, '--input', actigraph, '--ID', unique_id, tag,
         '--output_dir', OUTPUT_DIR],
        capture_output=True, text=True,
    )
    return filename, result.returncode, result.stdout, result.stderr


if __name__ == '__main__':
    print(f"Found {len(list_actigraph)} files — running with {N_WORKERS} workers")
    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(process_one, i, f): f
                   for i, f in enumerate(list_actigraph)}
        for future in as_completed(futures):
            fname, rc, out, err = future.result()
            status = 'OK  ' if rc == 0 else 'FAIL'
            print(f'[{status}] {fname}')
            if out: print(out, end='')
            if err: print(err, end='')
