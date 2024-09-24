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
            *map(annotation.get, ['onset', 'duration', 'PT1-1'])):
        if not pd.isnull(label):
            intresserad[(onset, duration)] = label

    upper = max(int(audio_len) - (STRIDE - 1), 1)
    label_timelist = [] # label_timelist是记录10s片段当中的label对应的onset和offset
    for curr in range(0, upper, STRIDE):
        start, end = curr, curr + WINDOW_SIZE
        new_filename = save_path.replace(".wav", f"_{start}_{end}.wav")
        # 保存切好的片段audio
        start_idx, end_idx = int(start*TARGET_SR), int(end*TARGET_SR)
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

def get_eval_durations(eval_meta_dir, max_len=3600):
    import soundfile as sf
    # generate duration tsv for eval data
    eval_meta = pd.read_csv(eval_meta_dir+"/eval.tsv", delimiter="\t")
    file_list = pd.unique(eval_meta["filename"].values)
    durations = []
    for file in file_list:
        wav, fs = sf.read(file)
        durations.append(min(len(wav) / fs, max_len))
    duration_df = pd.DataFrame({"filename": file_list, "duration": durations})
    duration_df.to_csv(eval_meta_dir+"/eval_durations.tsv", index=False, sep="\t")

def gen_eval_tsv():
    all_annotation_count = 0
    audio_dir = "/20A021/ccomhuqin/data/eval"
    eval_meta_dir = "/20A021/ccomhuqin/meta/eval"
    df = pd.DataFrame([], columns=["filename", "onset", "offset", "event_label"])


    for fullpath in Path(audio_dir).glob('**/*.wav'):
        audio_path = str(fullpath)
        csv_path = audio_path.replace(".wav", ".csv")
        annotation = pd.read_csv(csv_path)
        item_df = annotation[annotation['PT1-1'].notnull()]
        all_annotation_count += item_df.shape[0]
        print(f"current file {audio_path}, has {item_df.shape[0]} items. Total: {all_annotation_count} items.")
        item_dict ={"filename": np.full(item_df.shape[0], audio_path),
               "onset": item_df['onset'].values,
               "offset": item_df['onset'].values+item_df['duration'].values,
               "event_label": item_df['PT1-1'].values}
        df = pd.concat([df, pd.DataFrame(item_dict)], ignore_index=True)
    df.to_csv(eval_meta_dir+"/eval.tsv", index=False, sep="\t")

    get_eval_durations(eval_meta_dir, max_len=3600) # 对于不切割的audio，计算duration时不用考虑最大值，这里用1个小时作为最大值

def gen_eval_seg_tsv():
    generate_segment_dataset("/20A021/ccomhuqin/data/eval",
                             "/20A021/ccomhuqin_seg/data/eval",
                             "/20A021/ccomhuqin_seg/meta/eval/eval.tsv")
    get_eval_durations("/20A021/ccomhuqin_seg/meta/eval", max_len=10)

if __name__ == "__main__":
    STRIDE = 5
    WINDOW_SIZE = 10
    TARGET_SR = 16000
    ori_data_dir = "/20A021/ccomhuqin/data"
    seg_data_dir = "/20A021/ccomhuqin_seg/data"
    meta_dir = "/20A021/ccomhuqin/meta"
    # generate_segment_dataset(ori_audio_dir=ori_data_dir + "/train",
    #                          seg_audio_dir=seg_data_dir+"/train",
    #                          save_meta=meta_dir+"/train/train.tsv")
    gen_eval_seg_tsv()