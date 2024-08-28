from audiossl.methods.atstframe.downstream.utils_psds_eval.evaluation import compute_per_intersection_macro_f1
import pandas as pd
import numpy as np
import os

def generate_pred_tsv():
    pred_path = "/20A021/compare_with/yolo/kfold0411_fold1_all"
    df = pd.DataFrame([], columns=["filename", "onset", "offset", "event_label", "conf"])
    for csv_path in os.listdir(pred_path):
        if not csv_path.endswith('.csv'):
            print(f"{csv_path} is not csv file.")
            continue
        full_path = os.path.join(pred_path, csv_path)
        item_df = pd.read_csv(full_path)
        audio_file = full_path.replace(".csv", ".wav")
        item_dict ={
               "filename": np.full(item_df.shape[0], audio_file),
               "onset": item_df['onset'].values,
               "offset": item_df['offset'].values,
               "event_label": item_df['label'].values,
               "conf": item_df['conf'].values
        }
        df = pd.concat([df, pd.DataFrame(item_dict)], ignore_index=True)
    df.to_csv(pred_path+"/all.tsv", index=False, sep="\t")

def compute_metrics():
    pred_path = "/20A021/compare_with/yolo/kfold0411_fold1_all"
    pred_df = pd.read_csv(pred_path+"/all.tsv", sep="\t")

    test_tsv = "/20A021/ccomhuqin_seg/meta/eval/eval_rm_intersect.tsv"
    test_dur = "/20A021/ccomhuqin_seg/meta/eval/eval_durations.tsv"

    intersection_f1_macro = compute_per_intersection_macro_f1(
        {"0.5": pred_df}, test_tsv, test_dur
    )
    print("Intersection F1:", intersection_f1_macro)

if __name__ == "__main__":
    compute_metrics()

