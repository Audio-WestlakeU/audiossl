from audiossl.methods.atstframe.downstream.utils_psds_eval.evaluation import compute_per_intersection_macro_f1, \
    compute_per_intersection_metrics
import pandas as pd
import numpy as np
import os

def generate_pred_tsv_yolo():
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

def generate_pred_tsv_mert(pred_path):
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
               "onset": item_df['start'].values,
               "offset": item_df['end'].values,
               "event_label": item_df['label'].values
        }
        df = pd.concat([df, pd.DataFrame(item_dict)], ignore_index=True)
    df.to_csv(pred_path+"/all.tsv", index=False, sep="\t")

def match_filename_with_gt(pred_path, gt_path = "/20A021/ccomhuqin/data/eval/"):
    pred_df = pd.read_csv(pred_path+"/all.tsv", sep="\t")
    pred_df['filename'] = pred_df['filename'].str.replace("/banhu_hbz", "/media/banhu_hbz")
    pred_df.to_csv(pred_path+"/all.tsv", index=False, sep="\t")
    #pred_df.replace("Huangjiangqin", "", regex=True)

def remove_DTG(pred_tsv_path):
    pred_df = pd.read_csv(pred_tsv_path)
    df = pred_df[pred_df.event_label != "DTG"]
    print("Remove label entries: ", pred_df.shape[0] - df.shape[0])
    df.to_csv(pred_tsv_path.replace('.csv', '_remove_DTG.csv'), index=False)

def compute_metrics(threshold, pred_csv, save_path=None):
    import json
    pred_df = pd.read_csv(pred_csv)


    # threshold is different for different models. Choose 0.25 for yolo
    # Mert use single-label cross_entropy prediction loss, no threshold.
    psds_metrics_mean, p_per_class_th, r_per_class_th, f_per_class_th = compute_per_intersection_metrics(
        {threshold: pred_df}, test_tsv, test_dur
    )
    if save_path is not None:
        with open(save_path + f'/results_metrics_{threshold}.txt', 'w') as data:
            data.writelines("Mean_metrics:")
            data.write(json.dumps(psds_metrics_mean))
            data.write(json.dumps(psds_metrics_mean))
            data.writelines("Precision_per_class:")
            data.write(json.dumps(p_per_class_th, allow_nan=True))
            data.writelines("Recall_per_class:")
            data.write(json.dumps(r_per_class_th, allow_nan=True))
            data.writelines("Fscore_per_class:")
            data.write(json.dumps(f_per_class_th, allow_nan=True))


if __name__ == "__main__":
    #generate_pred_tsv_mert(pred_path = "/20A021/compare_with/mert/results_1227")
    #----------------ATST-----------------------
    test_tsv = "/20A021/ccomhuqin_seg/meta/eval/eval_rm_intersect.tsv"
    test_dur = "/20A021/ccomhuqin_seg/meta/eval/eval_durations.tsv"
    metrics_dir = "/20A021/ccomhuqin_seg/save_path/finetune0922/metrics_test/"
    compute_metrics(threshold=0.5, pred_csv=metrics_dir + "pred_0.5.csv", save_path=metrics_dir)

    #---------------MERT-------------------
    # test_tsv = "/20A021/ccomhuqin/meta/eval/eval_rm_intersect.tsv"
    # test_dur = "/20A021/ccomhuqin/meta/eval/eval_durations.tsv"
    # metrics_dir = "/20A021/compare_with/mert/results_1227/"
    # compute_metrics(threshold=0.5, pred_csv=metrics_dir + "remove_DTG_all.csv")





