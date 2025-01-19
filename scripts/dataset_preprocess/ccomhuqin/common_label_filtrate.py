import os
import pandas as pd
import numpy as np
from argparse import ArgumentParser

'''
在Audioset_strong数据集，只有train和eval，没有validation set
胡琴数据集从train拿出了部分作为validation, eval就是test数据集
'''
def old_implementation(seg_meta_path, labels_to_remove, single_eval_meta_tsv):
    '''
    这个操作有问题，计算common_labels之后，只要有一个样本的label不在common_label里，整个样本都删除
    '''
    os.chdir(seg_meta_path)

    train_meta = pd.read_csv("./train/train.tsv", sep="\t")
    val_meta = pd.read_csv("./val/val.tsv", sep="\t")
    eval_inference_meta = pd.read_csv("./eval/eval.tsv", sep="\t")
    eval_gt_meta = pd.read_csv(single_eval_meta_tsv, sep="\t")

    print("Total train files(seg):", len(pd.unique(train_meta["filename"])))
    print("Total val files(seg):", len(pd.unique(val_meta["filename"])))
    print("Total eval files(seg):", len(pd.unique(eval_inference_meta["filename"])))
    print("Total eval files(complete, ground-truth):", len(pd.unique(eval_gt_meta["filename"])))
    train_labels = pd.unique(train_meta["event_label"])
    val_labels = pd.unique(val_meta["event_label"])
    eval_labels = pd.unique(eval_inference_meta["event_label"])

    print("Total train labels:", len(set(train_labels)))
    print("Total val labels:", len(set(val_labels)))
    print("Total eval labels:", len(set(eval_labels)))
    train_val_common_labels = list(set(train_labels).intersection(set(val_labels)))
    common_labels = list(set(train_labels).intersection(set(eval_labels)))

    assert train_val_common_labels.sort() == common_labels.sort()

    for label in labels_to_remove:
        common_labels.remove(label)

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
    eval_rm_files = np.unique([filename for filename, label in zip(eval_inference_meta["filename"], eval_inference_meta["event_label"]) if
                               label not in common_labels])
    eval_gt_rm_files = np.unique(
        [filename for filename, label in zip(eval_gt_meta["filename"], eval_gt_meta["event_label"]) if
         label not in common_labels])
    print("Train remove: ", len(train_rm_files), " Val remove: ", len(val_rm_files), " Eval remove: ", len(eval_rm_files))
    print("Eval ground-truth(complete) remove: ", len(eval_gt_rm_files))

    train_label_mask = np.array([True if x not in train_rm_files else False for x in train_meta["filename"]])
    val_label_mask = np.array([True if x not in val_rm_files else False for x in val_meta["filename"]])
    eval_label_mask = np.array([True if x not in eval_rm_files else False for x in eval_inference_meta["filename"]])
    eval_gt_label_mask = np.array([True if x not in eval_gt_rm_files else False for x in eval_gt_meta["filename"]])
    train_meta_filt = train_meta[train_label_mask]
    val_meta_filt = val_meta[val_label_mask]
    eval_meta_filt = eval_inference_meta[eval_label_mask]
    eval_gt_meta_filt = eval_gt_meta[eval_gt_label_mask]

    print("Total train files after filtering:", len(pd.unique(train_meta_filt["filename"])))
    print("Total val files after filtering:", len(pd.unique(val_meta_filt["filename"])))
    print("Total eval files after filtering:", len(pd.unique(eval_meta_filt["filename"])))
    print("Total eval files(complete) after filtering:", len(pd.unique(eval_gt_meta_filt["filename"])))

    train_meta_filt.to_csv("./train/train_common.tsv", index=False, sep="\t")
    val_meta_filt.to_csv("./val/val_common.tsv", index=False, sep="\t")
    eval_meta_filt.to_csv("./eval/eval_common.tsv", index=False, sep="\t")
    eval_gt_meta_filt.to_csv(single_eval_meta_tsv.replace('.tsv', '_common.tsv'), index=False, sep="\t")


def main(train_meta_tsv, val_meta_tsv, eval_meta_tsv, eval_gt_meta_tsv, save_common_labels_txt):
    train_meta = pd.read_csv(train_meta_tsv, sep="\t")
    val_meta = pd.read_csv(val_meta_tsv, sep="\t")
    eval_meta = pd.read_csv(eval_meta_tsv, sep="\t")

    train_labels = pd.unique(train_meta["event_label"])
    val_labels = pd.unique(val_meta["event_label"])
    eval_labels = pd.unique(eval_meta["event_label"])
    train_val_common_labels = list(set(train_labels).intersection(set(val_labels)))
    common_labels = list(set(train_labels).intersection(set(eval_labels)))
    assert train_val_common_labels.sort() == common_labels.sort()

    for label in labels_to_remove:
        common_labels.remove(label)

    print("Common labels:", common_labels)
    with open(save_common_labels_txt, "w") as f:
        f.write("\n".join(common_labels))

    for meta_tsv in [train_meta_tsv, val_meta_tsv, eval_meta_tsv, eval_gt_meta_tsv]:
        meta = pd.read_csv(meta_tsv, sep="\t")
        for idx, row in meta.iterrows():
            label = row['event_label']
            if label not in common_labels:
                print(f"Drop {label} in {row['filename']}")
                meta = meta.drop(idx)
        meta.to_csv(meta_tsv.replace('.tsv', '_common.tsv'), index=False, sep='\t')


if __name__ == "__main__":
    meta_folder = "meta1-1"
    meta_dir = f"/20A021/ccomhuqin_seg/{meta_folder}"

    labels_to_remove = {'DTG', 'Pizz'}
    single_eval_meta_tsv = f"/20A021/ccomhuqin/{meta_folder}/eval/eval.tsv" # 这个文件是为了inference之后，拼接回原始测试集的annotation

    main(train_meta_tsv=meta_dir + '/train/train.tsv',
         val_meta_tsv=meta_dir+'/val/val.tsv',
         eval_meta_tsv=meta_dir+'/eval/eval.tsv',
         eval_gt_meta_tsv=single_eval_meta_tsv,
         save_common_labels_txt=meta_dir+'/common_labels.txt')