# This file is to clean up the intersections (of the same class) in the train/eval meta files.

# This file is the last step of the data cleaning, assuming that we have three files: train/val/eval
import os

import pandas as pd
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

parser = ArgumentParser("Data Preprocess")
parser.add_argument("--root_path", type=str, required=True)
args = parser.parse_args()

root_path = args.root_path
os.chdir(root_path) 

def process_onset_offset(onset, offset):
    onset_list, offset_list = [], []
    wide_onset = onset[0]
    wide_offset = offset[0]
    for i in range(len(onset)):
        this_onset = onset[i]
        this_offset = offset[i]
        if wide_offset > this_onset:
            if this_offset > wide_offset:
                wide_offset = this_offset
        else:
            onset_list.append(wide_onset)
            offset_list.append(wide_offset)
            wide_onset = this_onset
            wide_offset = this_offset
    else:
        onset_list.append(wide_onset)
        offset_list.append(wide_offset)
    return np.array(onset_list), np.array(offset_list)

def rm_intersec(df):
    all_files = pd.unique(df["filename"].values)
    return_df = pd.DataFrame([], columns=["filename", "onset", "offset", "event_label"])
    for file in tqdm(all_files):
        file_df = df[df["filename"] == file]
        all_events = pd.unique(file_df["event_label"].values)
        onset = file_df["onset"].values
        offset = file_df["offset"].values
        for event in all_events:
            event_mask = file_df["event_label"].values == event
            event_onsets = onset[event_mask]
            event_offsets = offset[event_mask]
            event_onsets, event_offsets = process_onset_offset(event_onsets, event_offsets)
            return_df = return_df.append(pd.DataFrame({"filename": [file] * len(event_onsets), "onset": event_onsets, "offset": event_offsets, "event_label": [event] * len(event_onsets)}))
    return return_df

train_df = pd.read_csv("./train/train_common.tsv", delimiter="\t")
eval_df = pd.read_csv("./eval/eval_common.tsv", delimiter="\t")

train_new = rm_intersec(train_df)
train_new.to_csv("./train/train_rm_intersect.tsv", index=False, sep="\t")
eval_new = rm_intersec(eval_df)
eval_new.to_csv("./eval/eval_rm_intersect.tsv", index=False, sep="\t")