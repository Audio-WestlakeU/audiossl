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

def generate_pred_tsv_atst(pred_path, save_path):
    df_all = pd.DataFrame([], columns=["filename", "onset", "offset", "event_label"])
    for csv_path in os.listdir(pred_path):
        csv_path = os.path.join(pred_path, csv_path)
        if not csv_path.endswith('.csv'):
            print(f"{csv_path} is not csv file.")
            continue
        df_all = pd.concat([df_all, pd.read_csv(csv_path)], ignore_index=True)
    df_all.to_csv(save_path+"/pred_all.csv", index=False)

def match_filename_with_gt(pred_path, gt_path = "/20A021/ccomhuqin/data/eval/"):
    pred_df = pd.read_csv(pred_path+"/all.tsv", sep="\t")
    pred_df['filename'] = pred_df['filename'].split_str.replace("/banhu_hbz", "/media/banhu_hbz")
    pred_df.to_csv(pred_path+"/all.tsv", index=False, sep="\t")
    #pred_df.replace("Huangjiangqin", "", regex=True)

def remove_DTG(pred_tsv_path):
    pred_df = pd.read_csv(pred_tsv_path)
    df = pred_df[pred_df.event_label != "DTG"]
    print("Remove label entries: ", pred_df.shape[0] - df.shape[0])
    df.to_csv(pred_tsv_path.replace('.csv', '_remove_DTG.csv'), index=False)

def compute_metrics(threshold, pred_csv, gt_tsv, gt_dur, save_path=None):
    import json
    pred_df = pd.read_csv(pred_csv)

    # threshold is different for different models. Choose 0.25 for yolo
    # Mert use single-label cross_entropy prediction loss, no threshold.
    psds_metrics_mean, p_per_class_th, r_per_class_th, f_per_class_th = compute_per_intersection_metrics(
        {threshold: pred_df}, gt_tsv, gt_dur
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

def rename_gt_atst(test_tsv, test_dur, save_test_tsv, save_test_dur):
    df = pd.read_csv(test_tsv, sep="\t")
    df['filename'] = df['filename'].apply(lambda x: x.split('/')[-1].split('.')[0])
    df.to_csv(save_test_tsv, index=False, sep='\t')

    df = pd.read_csv(test_dur, sep="\t")
    df['filename'] = df['filename'].apply(lambda x: x.split('/')[-1].split('.')[0])
    df.to_csv(save_test_dur, index=False, sep='\t')

def check_pred_match_gt(pred_csv, gt_tsv, gt_duration_tsv):
    pred_filenames = set(pd.read_csv(pred_csv)['filename'])
    gt_filenames = set(pd.read_csv(gt_tsv, sep='\t')['filename'])
    gt_duration_filenames = set(pd.read_csv(gt_duration_tsv, sep='\t')['filename'])
    assert pred_filenames == gt_filenames
    assert gt_filenames == gt_duration_filenames

if __name__ == "__main__":
    # ori_test_tsv = "/20A021/ccomhuqin/meta1-1/eval/eval_rm_intersect.tsv"
    # ori_test_dur = "/20A021/ccomhuqin/meta1-1/eval/eval_durations.tsv"
    gt_test_tsv = "/20A021/finetune_music_dataset/exp/audiossl/1-1/eval_rm_intersect.tsv"
    gt_test_dur = "/20A021/finetune_music_dataset/exp/audiossl/1-1/eval_durations.tsv"
    #rename_gt_atst(ori_test_tsv, ori_test_dur, gt_test_tsv, gt_test_dur) 只跑一遍

    #----------------- ATST-------------
    # atst_metrics_dir = "/20A021/finetune_music_dataset/exp/audiossl/1-1/freeze/0111/metrics_test/"
    # predictions_dir = os.path.join(metrics_dir, 'predictions')
    # generate_pred_tsv_atst(pred_path=predictions_dir, save_path=metrics_dir)
    #
    # check_pred_match_gt(pred_csv=metrics_dir+"pred_all.csv", gt_tsv=gt_test_tsv, gt_duration_tsv=gt_test_dur)
    # compute_metrics(threshold=0, pred_csv=metrics_dir+"pred_all.csv",
    #                 gt_tsv=gt_test_tsv, gt_dur=gt_test_dur,
    #                 save_path=metrics_dir)

    #---------------MERT-------------------
    mert_metrics_dir = '/20A021/compare_with/mert/1-1/0113/results/'
    predictions_dir = os.path.join(mert_metrics_dir, 'predictions')
    generate_pred_tsv_atst(pred_path=predictions_dir, save_path=mert_metrics_dir)

    check_pred_match_gt(pred_csv=mert_metrics_dir + "pred_all.csv", gt_tsv=gt_test_tsv, gt_duration_tsv=gt_test_dur)
    compute_metrics(threshold=0, pred_csv=mert_metrics_dir + "pred_all.csv",
                    gt_tsv=gt_test_tsv, gt_dur=gt_test_dur,
                    save_path=mert_metrics_dir)







