# Copy and adapt from audioset.py
# Different from music_dataset.py, this exactly copies audioset.py to create lmdb files and perform resampling operations in preprocessing.

from pathlib import Path
import torchaudio
import os
import time
from dataset2lmdb import folder2lmdb


def parse_csv(csv_file,lbl_map,head_lines=3):
    dict = {}
    csv_lines = [ x.strip() for x in open(csv_file,"r").readlines()]
    for line in csv_lines[head_lines:]:
        line_list = line.split(",")
        id,label = line_list[0],line_list[3:]
        label = [x.strip(" ").strip('\"') for x in label]
        for x in label:
            assert(x in lbl_map.keys())
        dict[id] = label
    return dict


def process(audio_path,dict):
    from tqdm import tqdm
    csv_output = "files,labels,ext\n"
    for path in tqdm(Path(audio_path).rglob('*.[wav mp3 flac AAC]*')):
        size = os.stat(path).st_size
        if  size < 500:
            print(path,size)
            continue
        base = path.name.split(".")[0][1:]
        label = dict[base]
        csv_output += "{},\"{}\",{}\n".format(path,",".join(label),base)
    return csv_output


if __name__ == "__main__":
    tsv_path = "/20A021/dataset_from_dyl/manifest_ub/segment_random_sample"
    lmdb_path = "/20A021/dataset_from_dyl/manifest_ub/lmdb"

    folder2lmdb(tsv_path,"train", lmdb_path, max_num=400000, write_frequency=10000, segment=10, sr=24000)
