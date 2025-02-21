import csv
import os
import shutil

import pandas as pd
import librosa
import torch
import torchaudio
import numpy as np
from pathlib import Path
from tqdm import tqdm

DATA_DIR = {"train": "ccom/train",
            "val": "ccom/val",
            "test": "ccom/test"}

EmptyToken = 'NA'


def generate_segment_dataset(ori_audio_dir, seg_audio_dir, save_meta):
    all_label_list = []
    if not os.path.exists(ori_audio_dir):
        raise FileNotFoundError
    else:
        for fullpath in Path(ori_audio_dir).glob('**/*.wav'):
            audio_path = str(fullpath)
            file_path = os.path.split(fullpath)[0]
            audio_file = os.path.split(fullpath)[1]
            save_path = file_path.replace(ori_audio_dir, seg_audio_dir)
            Path(save_path).mkdir(parents=True, exist_ok=True)
            all_label_list += split2segments(audio_path, os.path.join(save_path, audio_file))
    with open(save_meta, "w") as f_output:
        tsv_output = csv.writer(f_output, delimiter='\t')
        tsv_output.writerow(["filename", "onset", "offset", "event_label"])
        tsv_output.writerows(all_label_list)


def split2segments(audio_path, save_path):
    csv_path = audio_path.replace(".wav", ".csv")
    annotation = pd.read_csv(csv_path)
    audio_len = librosa.get_duration(path=audio_path)
    waveform, sr = torchaudio.load(audio_path, normalize=True)
    waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)
    waveform_mono = torch.mean(waveform, dim=0, keepdim=True)  # 双声道/立体声转成单声道
    print("converting", audio_path)

    intresserad = dict()
    for (onset, duration, label) in zip(
            *map(annotation.get, ['onset', 'duration', event_label_col])):
        if not pd.isnull(label):
            intresserad[(onset, duration)] = label

    upper = max(int(audio_len) - (STRIDE - 1), 1)
    label_timelist = []  # label_timelist是记录10s片段当中的label对应的onset和offset
    for curr in range(0, upper, STRIDE):
        start, end = curr, curr + WINDOW_SIZE
        new_filename = save_path.replace(".wav", f"_{start}_{end}.wav")
        # 保存切好的片段audio
        start_idx, end_idx = int(start * TARGET_SR), int(end * TARGET_SR)
        clip = waveform_mono[:, start_idx: end_idx]
        if os.path.exists(new_filename):
            raise Exception(f'{new_filename} already exists!')
        torchaudio.save(new_filename, clip, TARGET_SR)
        for onset, duration in intresserad:
            label = intresserad[(onset, duration)]
            # 标记的PT有任意一段在当前clip内，都要标记对应的frame
            offset = onset + duration
            if offset <= start or end <= onset:
                # no overlap
                continue
            elif onset <= start <= offset:
                onset = start
            elif onset <= end <= offset:
                offset = end
            elif onset <= start <= end <= offset:
                onset, offset = start, end
            new_onset, new_offset = onset - start, offset - start
            label_timelist.append([new_filename, new_onset, new_offset, label])
    return label_timelist


def check_onset_offset(min_label_sec, annotation_tsv):
    print(annotation_tsv)
    df = pd.read_csv(annotation_tsv, sep='\t')
    print(f'original len(df) is {len(df)}')
    for index, row in df.iterrows():
        onset_sec, offset_sec = round(row['onset'], 3), round(row['offset'], 3)
        # 先检查 onset<offset
        if onset_sec >= offset_sec:
            print('after round to 3, onset>offset', row)
            df = df.drop(index)
        else:
            # 再检查duration > min_label_sec
            duration = offset_sec - onset_sec
            if duration < min_label_sec:
                print(f'duration is less than {min_label_sec}s', duration, onset_sec, offset_sec,
                      row['event_label'], row['filename'])
                df = df.drop(index)
    print(f'after removal len(df) is {len(df)}')
    print(f'Save to {annotation_tsv}')
    df.to_csv(annotation_tsv, sep='\t', index=False)


def get_durations(meta_tsv, save_tsv):
    import soundfile as sf
    eval_meta = pd.read_csv(meta_tsv, delimiter="\t")
    file_list = pd.unique(eval_meta["filename"].values)
    durations = []
    for file in file_list:
        wav, fs = sf.read(file)
        durations.append(len(wav) / fs)
    duration_df = pd.DataFrame({"filename": file_list, "duration": durations})
    duration_df.to_csv(save_tsv, index=False, sep="\t")


def gen_eval_tsv(eval_meta_dir):
    all_annotation_count = 0
    audio_dir = "/20A021/ccomhuqin/data/eval"

    df = pd.DataFrame(columns=["filename", "onset", "offset", "event_label"])

    for fullpath in Path(audio_dir).glob('**/*.wav'):
        audio_path = str(fullpath)
        csv_path = audio_path.replace(".wav", ".csv")
        annotation = pd.read_csv(csv_path)
        item_df = annotation[annotation[event_label_col].notnull()]
        all_annotation_count += item_df.shape[0]
        print(f"current file {audio_path}, has {item_df.shape[0]} items. Total: {all_annotation_count} items.")
        item_dict = {"filename": np.full(item_df.shape[0], audio_path),
                     "onset": item_df['onset'].values,
                     "offset": item_df['onset'].values + item_df['duration'].values,
                     "event_label": item_df[event_label_col].values}
        df = pd.concat([df, pd.DataFrame(item_dict)], ignore_index=True)
    df.to_csv(eval_meta_dir + "/eval.tsv", index=False, sep="\t")

    get_durations(eval_meta_dir + "/eval.tsv", eval_meta_dir + "/eval_durations.tsv")


def gen_eval_seg_tsv(ori_audio_dir, save_audio_dir, save_meta_dir):
    save_meta_tsv = save_meta_dir+"/eval.tsv"
    generate_segment_dataset(ori_audio_dir, save_audio_dir, save_meta_tsv)

    save_meta_duration_tsv = save_meta_dir + "/eval_duration.tsv"
    get_durations(save_meta_tsv, save_meta_duration_tsv)

def move_to_newfold():
    file_count = 0
    for i in range(5):
        dst_folder = f"/20A021/ccomhuqin/data/fold_{i + 1}"
        os.makedirs(dst_folder, exist_ok=True)
        df = pd.read_csv(f"/20A021/ccomhuqin/meta1-1/fold_{i + 1}.csv")
        for idx, row in df.iterrows():
            filename = row['filename']
            # 复制文件到目标文件夹，并保留元数据
            shutil.copy2(filename, dst_folder)
            shutil.copy2(filename.replace('.csv', '.wav'), dst_folder)
            print(f"Moved {filename} and wav to {dst_folder}")
            file_count += 2
    print('Copied files: ', file_count)

def gen_train_val_from_fold(seg_meta_dir):
    train_folder = os.path.join(seg_meta_dir, 'train')
    val_folder = os.path.join(seg_meta_dir, 'val')
    for i in range(5):
        # 验证集：当前fold
        val_tsv = seg_meta_dir + f"/fold_{i+1}.tsv"
        val_df = pd.read_csv(val_tsv, sep='\t')
        # 训练集：其他4个folds
        train_dfs = []
        for j in range(5):
            if j == i:
                continue
            fold_path = os.path.join(seg_meta_dir, f"fold_{j+1}.tsv")
            train_dfs.append(pd.read_csv(fold_path, sep='\t'))
        train_df = pd.concat(train_dfs, axis=0)

        # 检查train和valid没有重复的
        common_filenames = set(train_df['filename']).intersection(set(val_df['filename']))
        if common_filenames:
            print("train_df 和 val_df 之间存在重复的 filename:")
            print(common_filenames)
            exit(1)
        # 保存
        train_df.to_csv(os.path.join(train_folder, f"train_fold_{i + 1}.tsv"), sep='\t', index=False)
        val_df.to_csv(os.path.join(val_folder, f"val_fold_{i + 1}.tsv"), sep='\t', index=False)
        print(f"Saved to {train_folder} and {val_folder}")


if __name__ == "__main__":
    STRIDE = 5
    WINDOW_SIZE = 5
    TARGET_SR = 16000
    event_label_col = 'PT1-1'
    ori_data_dir = "/20A021/ccomhuqin/data"
    seg_data_dir = "/20A021/ccomhuqin_seg/16k_5/data"
    seg_meta_dir = "/20A021/ccomhuqin_seg/16k_5/meta1-1"

    # 1. 生成10s的训练数据和验证数据
    # for k in range(5):
        # save_meta = seg_meta_dir + f"/fold_{k+1}.tsv"
        # generate_segment_dataset(ori_audio_dir=ori_data_dir + f"/fold_{k + 1}",
        #                          seg_audio_dir=seg_data_dir + f"/fold_{k + 1}",
        #                          save_meta=save_meta)
        #get_durations(save_meta, seg_meta_dir + f"/{split_str}/{split_str}_duration.tsv")

    #2. 生成10s的测试数据，用于inference
    # gen_eval_seg_tsv(ori_audio_dir=ori_data_dir+"/eval",
    #                  save_audio_dir=seg_data_dir+"/eval",
    #                  save_meta_dir=seg_meta_dir+"/eval")

    # 3. 生成仅测试数据的ground_truth标注，用于计算metrics
    # 这一步对于不同window_size或者sr都是一样的，只要测试原始数据不变，就不用再次生成。
    # gen_eval_tsv(eval_meta_dir="/20A021/ccomhuqin/meta1-1/eval")

    # 4. 检查所有的标注技法的时长，为了排除错误标注和裁剪的问题
    # min_label_sec = 0.04  # 据统计，最短的是DTG最短时长在0.04-0.05之间。其他小于0.04的大部分是有边缘裁剪，和手工错标。
    # for k in range(5):
    #     meta_tsv = seg_meta_dir + f"/fold_{k+1}.tsv"
    #     check_onset_offset(min_label_sec, meta_tsv)

    # 5. 把fold5个文件组合成5组train_valid
    gen_train_val_from_fold(seg_meta_dir=seg_meta_dir)


