import os
import pandas as pd
import numpy as np
from argparse import ArgumentParser

'''
在Audioset_strong数据集，只有train和eval，没有validation set
胡琴数据集从train拿出了部分作为validation, eval就是test数据集
'''
def main(meta_path):
    os.chdir(meta_path)

    train_meta = pd.read_csv("./train/train.tsv", sep="\t")
    val_meta = pd.read_csv("./val/val.tsv", sep="\t")
    eval_meta = pd.read_csv("./eval/eval.tsv", sep="\t")

    print("Total train files (in meta and we have):", len(pd.unique(train_meta["filename"])))
    print("Total val files (in meta and we have):", len(pd.unique(val_meta["filename"])))
    print("Total eval files (in meta and we have):", len(pd.unique(eval_meta["filename"])))
    train_labels = pd.unique(train_meta["event_label"])
    val_labels = pd.unique(val_meta["event_label"])
    eval_labels = pd.unique(eval_meta["event_label"])

    print("Total train labels:", len(set(train_labels)))
    print("Total val labels:", len(set(val_labels)))
    print("Total eval labels:", len(set(eval_labels)))
    train_val_common_labels = list(set(train_labels).intersection(set(val_labels)))
    common_labels = list(set(train_labels).intersection(set(eval_labels)))

    assert train_val_common_labels.sort() == common_labels.sort()

    common_labels.remove(LABEL_to_remove)

    print("Total common labels:", len(common_labels))
    with open("./common_labels.txt", "w") as f:
        f.write("\n".join(common_labels))
    # print("Disjoint train labels:", [x for x in set(train_labels) if x not in common_labels])
    # print("Disjoint eval labels:", [x for x in set(eval_labels) if x not in common_labels])
    train_rm_files = np.unique(
        [filename for filename, label in zip(train_meta["filename"], train_meta["event_label"]) if
         label not in common_labels])
    val_rm_files = np.unique(
        [filename for filename, label in zip(val_meta["filename"], val_meta["event_label"]) if
         label not in common_labels])
    eval_rm_files = np.unique([filename for filename, label in zip(eval_meta["filename"], eval_meta["event_label"]) if
                               label not in common_labels])
    print("Train remove: ", len(train_rm_files), " Val remove: ", len(val_rm_files), " Eval remove: ", len(eval_rm_files))

    train_label_mask = np.array([True if x not in train_rm_files else False for x in train_meta["filename"]])
    val_label_mask = np.array([True if x not in val_rm_files else False for x in val_meta["filename"]])
    eval_label_mask = np.array([True if x not in eval_rm_files else False for x in eval_meta["filename"]])
    train_meta_filt = train_meta[train_label_mask]
    val_meta_filt = val_meta[val_label_mask]
    eval_meta_filt = eval_meta[eval_label_mask]

    print("Total train files after filtering:", len(pd.unique(train_meta_filt["filename"])))
    print("Total val files after filtering:", len(pd.unique(val_meta_filt["filename"])))
    print("Total eval files after filtering:", len(pd.unique(eval_meta_filt["filename"])))

    train_meta_filt.to_csv("./train/train_common.tsv", index=False, sep="\t")
    val_meta_filt.to_csv("./val/val_common.tsv", index=False, sep="\t")
    eval_meta_filt.to_csv("./eval/eval_common.tsv", index=False, sep="\t")


if __name__ == "__main__":
    LABEL_to_remove = "DTG"
    meta_path = "/20A021/ccomhuqin_seg/meta"
    main(meta_path)