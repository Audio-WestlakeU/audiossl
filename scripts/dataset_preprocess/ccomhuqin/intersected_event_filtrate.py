# This file is to clean up the intersections (of the same class) in the train/eval meta files.

# This file is the last step of the data cleaning, assuming that we have three files: train/val/eval
import os
import warnings

import pandas as pd
import numpy as np
from tqdm import tqdm


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
    warnings.warn(
        "实验显示使用rm_intersect训练得到的结果是抛弓没有metrics，因此在这里提醒不要使用.",
        DeprecationWarning,
        stacklevel=2  # 确保警告指向调用者，而不是当前函数
    )
    print("This is a deprecated function.")
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
            #row_list.append([[file] * len(event_onsets), event_onsets, event_offsets, [event] * len(event_onsets)])
            return_df = pd.concat([return_df, pd.DataFrame(
                {"filename": [file] * len(event_onsets),
                 "onset": event_onsets,
                 "offset": event_offsets,
                 "event_label": [event] * len(event_onsets)
                 })])
    return return_df

def calculate_label_duration(meta_dir="/20A021/ccomhuqin_seg/meta/train", meta_tsv='train_common.tsv'):
    train_duration_df = pd.read_csv(meta_dir+'/train_duration.tsv', sep="\t")
    total_duration = train_duration_df['duration'].sum()
    print(f'Add durations of all audio files in {meta_dir}: {total_duration/60.0} minutes')

    train_meta = pd.read_csv(os.path.join(meta_dir,meta_tsv), sep="\t")
    class_dict = dict()
    for index, row in train_meta.iterrows():
        label = row['event_label']
        label_duration = row['offset'] - row['onset']
        if label in class_dict:
            class_dict[label] += label_duration
        else:
            class_dict[label] = label_duration
    row = []
    for key, value in class_dict.items():
        row.append([key, value, value/total_duration])
    df = pd.DataFrame(row, columns=['label', 'duration', 'ratio'])
    df.to_csv(meta_dir+'/label_duration.tsv', sep="\t", index=False)


def main(meta_path="/20A021/ccomhuqin_seg/meta"):
    os.chdir(meta_path)
    train_df = pd.read_csv("./train/train_common.tsv", delimiter="\t")
    val_df = pd.read_csv("./val/val_common.tsv", delimiter="\t")
    eval_df = pd.read_csv("./eval/eval_common.tsv", delimiter="\t")

    train_new = rm_intersec(train_df)
    diff = pd.concat([train_df, train_new]).drop_duplicates(keep=False)
    diff['filename'] = diff['filename'].apply(os.path.basename)
    print('------------------diff for train----------------')
    print(diff)
    print('------------------End of train----------------')
    train_new.to_csv("./train/train_rm_intersect.tsv", index=False, sep="\t")

    val_new = rm_intersec(val_df)
    diff = pd.concat([val_df, val_new]).drop_duplicates(keep=False)
    print('------------------diff for val----------------')
    print(diff)
    print('------------------End of val----------------')
    val_new.to_csv("./val/val_rm_intersect.tsv", index=False, sep="\t")

    eval_new = rm_intersec(eval_df)
    diff = pd.concat([eval_df, eval_new]).drop_duplicates(keep=False)
    print('------------------diff for eval----------------')
    print(diff)
    print('------------------End of eval----------------')
    eval_new.to_csv("./eval/eval_rm_intersect.tsv", index=False, sep="\t")

def rm_intersect_gt():
    meta_dir = "/20A021/ccomhuqin/meta1-1"
    eval_df = pd.read_csv(meta_dir+"/eval/eval_common.tsv", delimiter="\t")
    eval_new = rm_intersec(eval_df)
    eval_new.to_csv(meta_dir+"/eval/eval_rm_intersect.tsv", index=False, sep="\t")

if __name__ == "__main__":
    #main()
    #calculate_label_duration(meta_dir="/20A021/ccomhuqin_seg/meta/train", meta_tsv='train_common.tsv')

    # 只有ccomhuqin 测试集的gt文件需要去掉intersect label，原因是计算psds时，不允许相同类别的frame重叠
    rm_intersect_gt()
