import csv
import os
import pandas as pd
import librosa
import torch
import torchaudio
import numpy as np
from pydub import AudioSegment
from pathlib import Path

DATA_DIR = {"train": "ccom/train",
            "val": "ccom/val",
            "test": "ccom/test"}

EmptyToken = 'NA'

STRIDE = 5
WINDOW_SIZE = 10
TARGET_SR = 16000
def generate_dataset(ori_audio_dir, seg_audio_dir, save_meta):
    key = 0
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
            # 标记的PT有任意一段在当前5s clip内，都要标记对应的frame
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

def get_eval_durations():
    import soundfile as sf
    # generate duration tsv for eval data
    eval_meta = pd.read_csv(meta_dir+"/eval/eval.tsv", delimiter="\t")
    file_list = pd.unique(eval_meta["filename"].values)
    durations = []
    for file in file_list:
        wav, fs = sf.read(file)
        durations.append(min(10, len(wav) / fs))
    duration_df = pd.DataFrame({"filename": file_list, "duration": durations})
    duration_df.to_csv(meta_dir+"/eval/eval_durations.tsv", index=False, sep="\t")

if __name__ == "__main__":
    ori_data_dir = "/20A021/ccomhuqin/data"
    seg_data_dir = "/20A021/ccomhuqin_seg/data"
    meta_dir = "/20A021/ccomhuqin_seg/meta"
    generate_dataset(ori_audio_dir=ori_data_dir+"/eval",
                     seg_audio_dir=seg_data_dir+"/eval",
                      save_meta=meta_dir+"/eval/eval.tsv")
    get_eval_durations()