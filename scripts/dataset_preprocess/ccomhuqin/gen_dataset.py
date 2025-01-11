import csv
import os
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


def gen_eval_seg_tsv(meta_dir):
    ori_data_dir = "/20A021/ccomhuqin/data/eval"
    save_meta_tsv = meta_dir + "/eval/eval.tsv"
    save_data_dir = ori_data_dir.replace('ccomhuqin', 'ccomhuqin_seg')
    generate_segment_dataset(ori_data_dir, save_data_dir, save_meta_tsv)

    save_meta_duration_tsv = meta_dir + "/eval/eval_duration.tsv"
    get_durations(save_meta_tsv, save_meta_duration_tsv)


if __name__ == "__main__":
    STRIDE = 5
    WINDOW_SIZE = 10
    TARGET_SR = 16000
    event_label_col = 'PT1-1'
    ori_data_dir = "/20A021/ccomhuqin/data"
    seg_data_dir = "/20A021/ccomhuqin_seg/data"
    ori_meta_dir = "/20A021/ccomhuqin/meta1-1"
    seg_meta_dir = "/20A021/ccomhuqin_seg/meta1-1"

    # 1. 生成10s的训练数据和验证数据
    # for split_str in ['train', 'val']:
    #     save_meta = meta_dir + f"/{split_str}/{split_str}.tsv"
    #     generate_segment_dataset(ori_audio_dir=ori_data_dir + f"/{split_str}",
    #                              seg_audio_dir=seg_data_dir + f"/{split_str}",
    #                              save_meta=save_meta)
    #     get_durations(save_meta, meta_dir + f"/{split_str}/{split_str}_duration.tsv")

    #2. 生成10s的测试数据，用于inference
    # gen_eval_seg_tsv(meta_dir=seg_meta_dir)

    # 3. 生成仅测试数据的ground_truth标注，用于计算metrics
    #gen_eval_tsv(eval_meta_dir=ori_meta_dir + "/eval")

    # 4. 检查所有的标注技法的时长，为了排除错误标注和裁剪的问题
    min_label_sec = 0.04  # 据统计，最短的是DTG最短时长在0.04-0.05之间。其他小于0.04的大部分是有边缘裁剪，和手工错标。
    meta_tsvs = [seg_meta_dir+"/train/train.tsv",
                 seg_meta_dir+"/val/val.tsv",
                 seg_meta_dir+"/eval/eval.tsv",
                 ori_meta_dir+"/eval/eval.tsv"
                 ]
    for tsv in meta_tsvs:
        check_onset_offset(min_label_sec, tsv)
