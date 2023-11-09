import os
import pandas as pd
import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser("Data Preprocess")
parser.add_argument("--root_path", type=str, required=True)
args = parser.parse_args()

root_path = args.root_path
os.chdir(root_path) 

train_meta = pd.read_csv("./train/train.tsv",sep="\t")
eval_meta = pd.read_csv("./eval/eval.tsv",sep="\t")

print("Total train files (in meta and we have):", len(pd.unique(train_meta["filename"])))
print("Total eval files (in meta and we have):", len(pd.unique(eval_meta["filename"])))
train_labels = pd.unique(train_meta["event_label"])
eval_labels = pd.unique(eval_meta["event_label"])

print("Total train labels:", len(set(train_labels)))
print("Total eval labels:", len(set(eval_labels)))
common_labels = list(set(train_labels).intersection(set(eval_labels)))
print("Total common labels:", len(common_labels))
with open("./common_labels.txt", "w") as f:
    f.write("\n".join(common_labels))
# print("Disjoint train labels:", [x for x in set(train_labels) if x not in common_labels])
# print("Disjoint eval labels:", [x for x in set(eval_labels) if x not in common_labels])
train_rm_files = np.unique([filename for filename, label in zip(train_meta["filename"], train_meta["event_label"]) if label not in common_labels])
eval_rm_files = np.unique([filename for filename, label in zip(eval_meta["filename"], eval_meta["event_label"]) if label not in common_labels])
print("Train remove: ", len(train_rm_files), "Eval remove: ", len(eval_rm_files))

train_label_mask = np.array([True if x not in train_rm_files else False for x in train_meta["filename"] ])
eval_label_mask = np.array([True if x not in eval_rm_files else False for x in eval_meta["filename"] ])
train_meta_filt = train_meta[train_label_mask]
eval_meta_filt = eval_meta[eval_label_mask]

print("Total train files after filtering:", len(pd.unique(train_meta_filt["filename"])))
print("Total eval files after filtering:", len(pd.unique(eval_meta_filt["filename"])))

train_meta_filt.to_csv("./train/train_common.tsv", index=False, sep="\t")
eval_meta_filt.to_csv("./eval/eval_common.tsv", index=False, sep="\t")