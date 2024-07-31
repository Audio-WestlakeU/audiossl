# Copy and adapt from audioset.py
# Read all audio files and Create train.tsv, which will be used in music_train_base.yaml
# This is for training directly from audio files, not creating lmdb files

from pathlib import Path
import os
import csv
from tqdm import tqdm
import pandas as pd

def parse_csv(csv_file,lbl_map,head_lines=3):
    dict = {}
    csv_lines = [ x.strip() for x in open(csv_file,"r").readlines()]
    for line in csv_lines[head_lines:]:
        line_list = line.split(",")
        id,label = line_list[0],line_list[3:]
        label = [x.strip(" ").strip('\"') for x in label]
        for x in label:
            if x == "":
                continue
            assert(x in lbl_map.keys())
        dict[id] = label
    return dict

def process():
    tsv_output = [["files", "labels", "ext"]]
    all_files_count = 0
    for path in tqdm(Path(audio_path).rglob('*.[wav mp3 flac AAC]*')):
        all_files_count += 1
        size = os.stat(path).st_size
        if  size < 500:
            print(f"Filtered {path} for its size({size}) less than 500.")
            continue
        base = path.name.split(".")[0]
        # 将所有label hardcode成ALL
        label = 'ALL'
        tsv_output.append([path, label, base])
    print(f"All files in {audio_path}: {all_files_count}")
    return tsv_output
def preprocess():
    import os
    import json
    label_csv_file = os.path.join(csv_path, "class_labels_indices.csv")
    lbl_map = {}
    with open(label_csv_file, "r") as f:
        label_lines = [x.strip() for x in f.readlines()]
    for line in label_lines[1:]:
        line_list = line.split(",")
        id, label = line_list[0], line_list[1]
        lbl_map[label] = int(id)
    with open(os.path.join(manifest_path, "lbl_map.json"), "w") as f:
        json.dump(lbl_map, f)

    train_tsv_output = process()
    with open(os.path.join(manifest_path, "tr.tsv"), "w") as f:
        tsv_output = csv.writer(f, delimiter='\t')
        tsv_output.writerows(train_tsv_output)

# preprocess之后，将不同的tsv合并在一起
def concatTsvs():
    all_filenames = ["/20A021/dataset_from_dyl/train-50up/manifest_ub/tr.tsv",
                "/20A021/dataset_from_dyl/train-15to45/manifest_ub/tr.tsv"]
    save_path = "/20A021/dataset_from_dyl/"
    combined_file = pd.concat([pd.read_csv(f, sep='\t') for f in all_filenames])
    combined_file.to_csv(os.path.join(save_path, 'tr.tsv'), sep='\t', index=False)

if __name__ == "__main__":
    data_path = "/20A021/dataset_from_dyl/train-50up"
    audio_path = os.path.join(data_path, "audio")
    csv_path = os.path.join(data_path, "csv")
    manifest_path = os.path.join(data_path, "manifest_ub")

    preprocess()

