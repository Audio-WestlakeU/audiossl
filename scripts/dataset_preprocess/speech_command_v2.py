from pathlib import Path
import torchaudio
import os
import sys
import json
from dataset2lmdb import folder2lmdb

labels = [
"right",
"eight",
"cat",
"backward",
"learn",
"tree",
"bed",
"happy",
"go",
"dog",
"no",
"wow",
"follow",
"nine",
"left",
"stop",
"three",
"sheila",
"one",
"bird",
"zero",
"seven",
"up",
"visual",
"marvin",
"two",
"house",
"down",
"six",
"yes",
"on",
"five",
"forward",
"off",
"four"
]

def load_set(txt_file):
    lines = [x.strip() for x in open(txt_file,"r").readlines()]
    return lines

def process(path,valid_set,eval_set):
    train_csv= "files,labels,ext\n"
    valid_csv= "files,labels,ext\n"
    eval_csv= "files,labels,ext\n"
    for path in Path(path).rglob('*.wav'):
        
        parts = path.parts
        key = "/".join(parts[-2:])
        label = parts[-2]
        name = parts[-1]
        if label == '_background_noise_':
            continue
        if key in valid_set:
            valid_csv += "{},\"{}\",{}\n".format(path,label,name)
        elif key in eval_set:
            eval_csv += "{},\"{}\",{}\n".format(path,label,name)
        else:
            train_csv += "{},\"{}\",{}\n".format(path,label,name)

    return train_csv,valid_csv,eval_csv




if __name__ == "__main__":

    path = sys.argv[1]
    eval_set = load_set(os.path.join(path,"testing_list.txt"))
    valid_set = load_set(os.path.join(path,"validation_list.txt"))

    train_csv,valid_csv,eval_csv = process(path,valid_set,eval_set)
    manifest_path = os.path.join(path,"manifest")
    lmdb_path = os.path.join(path,"lmdb")
    os.makedirs(manifest_path,exist_ok=True)
    os.makedirs(lmdb_path,exist_ok=True)
    with open(os.path.join(manifest_path,"tr.csv"),"w") as f:
        f.write(train_csv)
    with open(os.path.join(manifest_path,"val.csv"),"w") as f:
        f.write(valid_csv)
    with open(os.path.join(manifest_path,"eval.csv"),"w") as f:
        f.write(eval_csv)

    lbl_map = {}
    for i,l in enumerate(labels):
        lbl_map[l]=i
    with open(os.path.join(manifest_path,"lbl_map.json"),"w") as f:
        json.dump(lbl_map,f)

    folder2lmdb(manifest_path,"train",lmdb_path)
    folder2lmdb(manifest_path,"valid",lmdb_path)
    folder2lmdb(manifest_path,"eval",lmdb_path)

