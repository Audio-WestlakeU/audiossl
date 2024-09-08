import os

import librosa
import pandas as pd
from tqdm import tqdm
import numpy as np

def compute_segments(tr_tsv, new_tr_tsv, randomly=True):
    r"""
    generate segments of WINDOW_SIZE 10s, with the stride 10s
    Args:
        tr_tsv: original TSV
        new_tr_tsv: new TSV with each segment as one row, plus the start_second and end_second
    """
    df_tr_tsv = pd.read_csv(tr_tsv, delimiter="\t")
    df_list = [] # will reset at fixed frequency
    total_count = 0
    if os.path.isfile(new_tr_tsv):
        print(f"Error: {new_tr_tsv} already exists! Delete it before writing.")
        return

    for index, row in tqdm(df_tr_tsv.iterrows()):
        file_path, label, ext = row["files"], row["labels"], row["ext"]
        audio_len = librosa.get_duration(path=file_path)
        upper = max(int(audio_len) - (STRIDE - 1), 1)

        # Two ways of generating segments:
        if randomly:
            # 1. generate randomly, from 0 to upper
            count = int(audio_len / STRIDE)
            start_index_list = np.sort(np.random.uniform(0, upper, count))
            for start_idx in start_index_list:
                df_list.append([file_path, start_idx, start_idx + WINDOW_SIZE, label, ext])
        else:
            # 2. generate sequentially, [0,10],[10,20], ...
            for curr in range(0, upper, STRIDE):
                start, end = curr, curr + WINDOW_SIZE
                df_list.append([file_path, start, end, label, ext])
        if int(index) % 1000 == 0:
            total_count += len(df_list)
            print(f"current index is {index}, {total_count} segments in total generated. Write to file: {new_tr_tsv}")
            with open("/20A021/dataset_from_dyl/manifest_ub/segment/output.txt", "w") as text_file:
                text_file.write("After segmentation, dataset len: %s" % total_count)
            df_to_concat = pd.DataFrame(data=df_list, columns=["files", "start_second", "end_second", "labels", "ext"])
            df_list = []  # remember to clean df_list
            if os.path.isfile(new_tr_tsv):
                df_new = pd.read_csv(new_tr_tsv, delimiter="\t")
                df_new = pd.concat([df_new, df_to_concat], ignore_index=True)
                assert(total_count == df_new.shape[0])
                df_new.to_csv(new_tr_tsv, sep="\t", index=False)
            else:
                df_to_concat.to_csv(new_tr_tsv, sep="\t", index=False)

def get_samplerate_distribution(tr_tsv):
    df_tr_tsv = pd.read_csv(tr_tsv, delimiter="\t")
    sr_dict = {}
    for index, row in tqdm(df_tr_tsv.iterrows()):
        file_path = row["files"]
        sr = librosa.get_samplerate(file_path)
        if sr in sr_dict.keys():
            sr_dict[sr] += 1
        else:
            sr_dict[sr] = 1
        if int(index) % 500 == 0:
            print(sr_dict)



if __name__ == "__main__":
    WINDOW_SIZE = 10 # anchor_len
    STRIDE = WINDOW_SIZE # No overlap
    tr_tsv = "/20A021/dataset_from_dyl/manifest_ub/truncate/tr.tsv"
    new_tr_tsv = "/20A021/dataset_from_dyl/manifest_ub/segment/tr_random.tsv"

    get_samplerate_distribution(tr_tsv)
    #compute_segments(tr_tsv, new_tr_tsv, randomly=True)