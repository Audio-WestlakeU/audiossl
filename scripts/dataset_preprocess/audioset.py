
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
    gpath=None
    csv_output = "files,labels,ext\n"
    for path in Path(audio_path).rglob('*.wav'):
        size = os.stat(path).st_size
        if  size < 500:
            print(path,size)
            continue
        base = path.name.split(".")[0][1:]
        label = dict[base]
        csv_output += "{},\"{}\",{}\n".format(path,",".join(label),base)
        gpath = path
    return csv_output


if __name__ == "__main__":
    import sys
    import os
    import json


    data_path = sys.argv[1:]

    audio_path = os.path.join(data_path,"audio")
    csv_path = os.path.join(data_path,"csv")

    manifest_path_b = os.path.join(data_path,"manifest_b")
    manifest_path_ub = os.path.join(data_path,"manifest_ub")

    lmdb_path_b = os.path.join(data_path,"manifest_b")
    lmdb_path_ub = os.path.join(data_path,"manifest_ub")

    label_csv_file = os.path.join(csv_path,"class_labels_indices.csv")
    lbl_map={}
    with open(label_csv_file,"r") as f:
        label_lines = [x.strip() for x in f.readlines()]
    for line in label_lines[1:]:
        line_list = line.split(",")
        id,label = line_list[0],line_list[1]
        lbl_map[label] = int(id)
        print(id,label)
    print(lbl_map)

    with open(os.path.join(manifest_path_b,"lbl_map.json"),"w") as f:
        json.dump(lbl_map,f)

    with open(os.path.join(manifest_path_ub,"lbl_map.json"),"w") as f:
        json.dump(lbl_map,f)



    unbalance_train_csv_file = os.path.join(csv_path,"unbalanced_train_segments.csv")
    balance_train_csv_file = os.path.join(csv_path,"balanced_train_segments.csv")
    eval_csv_file = os.path.join(csv_path,"eval_segments.csv")

    eval_dict = parse_csv(eval_csv_file,lbl_map)
    unb_train_dict = parse_csv(unbalance_train_csv_file,lbl_map)
    b_train_dict = parse_csv(balance_train_csv_file,lbl_map)

    eval_audio_path = os.path.join(audio_path,"eval_segments")
    eval_csv_output = process(eval_audio_path,eval_dict)

    b_train_audio_path = os.path.join(audio_path,"balanced_train_segments")
    b_train_csv_output = process(b_train_audio_path,b_train_dict)

    unb_train_audio_path = os.path.join(audio_path,"unbalanced_train_segments")
    unb_train_csv_output = process(unb_train_audio_path,unb_train_dict)

    with open(os.path.join(manifest_path_ub,"trn.csv"),"w") as f:
        f.write(unb_train_csv_output)
    with open(os.path.join(manifest_path_b,"trn.csv"),"w") as f:
        f.write(b_train_csv_output)
    with open(os.path.join(manifest_path_ub,"eval.csv"),"w") as f:
        f.write(eval_csv_output)
    with open(os.path.join(manifest_path_b,"eval.csv"),"w") as f:
        f.write(eval_csv_output)
    with open(os.path.join(manifest_path_ub,"valid.csv"),"w") as f:
        f.write(eval_csv_output)
    with open(os.path.join(manifest_path_b,"valid.csv"),"w") as f:
        f.write(eval_csv_output)
    
    folder2lmdb(manifest_path_ub,"train",lmdb_path_ub)
    folder2lmdb(manifest_path_ub,"valid",lmdb_path_ub)
    folder2lmdb(manifest_path_ub,"eval",lmdb_path_ub)

    folder2lmdb(manifest_path_b,"train",lmdb_path_b)
    folder2lmdb(manifest_path_b,"valid",lmdb_path_b)
    folder2lmdb(manifest_path_b,"eval",lmdb_path_b)