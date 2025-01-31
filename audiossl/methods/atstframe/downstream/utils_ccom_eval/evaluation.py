from audiossl.methods.atstframe.downstream.utils_psds_eval.evaluation import compute_per_intersection_macro_f1, \
    compute_per_intersection_metrics
import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score, f1_score, classification_report


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
        item_dict = {
            "filename": np.full(item_df.shape[0], audio_file),
            "onset": item_df['onset'].values,
            "offset": item_df['offset'].values,
            "event_label": item_df['label'].values,
            "conf": item_df['conf'].values
        }
        df = pd.concat([df, pd.DataFrame(item_dict)], ignore_index=True)
    df.to_csv(pred_path + "/all.tsv", index=False, sep="\t")


def generate_pred_tsv_atst(pred_path, save_path):
    df_all = pd.DataFrame([], columns=["filename", "onset", "offset", "event_label"])
    for csv_path in os.listdir(pred_path):
        csv_path = os.path.join(pred_path, csv_path)
        if not csv_path.endswith('.csv'):
            print(f"{csv_path} is not csv file.")
            continue
        df_all = pd.concat([df_all, pd.read_csv(csv_path)], ignore_index=True)
    df_all.to_csv(save_path + "/pred_all.csv", index=False)


def match_filename_with_gt(pred_path, gt_path="/20A021/ccomhuqin/data/eval/"):
    pred_df = pd.read_csv(pred_path + "/all.tsv", sep="\t")
    pred_df['filename'] = pred_df['filename'].split_str.replace("/banhu_hbz", "/media/banhu_hbz")
    pred_df.to_csv(pred_path + "/all.tsv", index=False, sep="\t")
    # pred_df.replace("Huangjiangqin", "", regex=True)


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
    # assert pred_filenames == gt_filenames
    assert gt_filenames == gt_duration_filenames


def compute_framewise_metrics(pred_csv, gt_tsv, dur_tsv, save_path):
    predictions = pd.read_csv(pred_csv)
    ground_truth = pd.read_csv(gt_tsv, sep="\t")
    gt_duration = pd.read_csv(dur_tsv, sep="\t")
    duration_dict = dict(zip(gt_duration["filename"], gt_duration["duration"]))
    grouped_predictions = predictions.groupby("filename")
    grouped_ground_truth = ground_truth.groupby("filename")
    filenames = predictions["filename"].unique()
    # 初始化全局帧级别标签
    all_frame_labels_gt = []
    all_frame_labels_pred = []

    # 遍历每个文件
    for filename in filenames:
        # 获取当前文件的 ground-truth 和 predictions
        ground_truth_events = grouped_ground_truth.get_group(filename)
        predicted_events = grouped_predictions.get_group(filename)
        # 生成当前文件的帧级别标签
        audio_duration = duration_dict[filename]
        frame_labels_gt = generate_frame_labels(ground_truth_events, audio_duration)
        frame_labels_pred = generate_frame_labels(predicted_events, audio_duration)
        # 拼接全局帧级别标签
        all_frame_labels_gt.extend(frame_labels_gt)
        all_frame_labels_pred.extend(frame_labels_pred)

    # 转换为 NumPy 数组
    all_frame_labels_gt = np.array(all_frame_labels_gt)
    all_frame_labels_pred = np.array(all_frame_labels_pred)

    # 使用 classification_report 打印每个类别的详细指标
    unique_labels = np.unique(np.concatenate([all_frame_labels_gt, all_frame_labels_pred]))
    report = classification_report(all_frame_labels_gt, all_frame_labels_pred, labels=unique_labels)

    # 计算全局 Accuracy
    global_accuracy = accuracy_score(all_frame_labels_gt, all_frame_labels_pred)
    # 计算全局 F1（需要将标签转换为数值）
    label_to_index = {label: i for i, label in enumerate(unique_labels)}
    all_frame_labels_gt_numeric = np.array([label_to_index[label] for label in all_frame_labels_gt])
    all_frame_labels_pred_numeric = np.array([label_to_index[label] for label in all_frame_labels_pred])
    global_f1 = f1_score(all_frame_labels_gt_numeric, all_frame_labels_pred_numeric, average="macro")
    print(f"Mean accuracy: {global_accuracy}. Macro F1: {global_f1}")
    # 保存到文件
    with open(os.path.join(save_path, "framewise_metrics.txt"), mode="w") as file:
        file.write(report)
        file.write("\n")
        file.write(f"Mean accuracy: {global_accuracy}. Macro F1: {global_f1}")
    print(f"Classification report and macro F1 saved to {save_path}")


def generate_frame_labels(events, audio_duration, frame_duration=0.04, overlap_threshold=0.5):
    # 生成每帧的起始和结束时间
    n_frames = int(audio_duration / frame_duration)
    frame_times = [(i * frame_duration, (i + 1) * frame_duration) for i in range(n_frames)]

    # 初始化帧级别标签
    frame_labels = np.full(n_frames, "NA", dtype=object)

    # 将事件转换为帧级别标签
    for _, row in events.iterrows():
        onset, offset, label = row["onset"], row["offset"], row["event_label"]
        for i, (frame_start, frame_end) in enumerate(frame_times):
            overlap = min(offset, frame_end) - max(onset, frame_start)
            if overlap > 0:  # 如果有重叠
                overlap_ratio = overlap / frame_duration  # 计算重叠比例
                if overlap_ratio > overlap_threshold:  # 如果重叠比例超过阈值
                    frame_labels[i] = label
    return frame_labels


def metrics_from_wandb(project_name=""):
    import wandb
    import numpy as np

    # 初始化 W&B API
    api = wandb.Api()
    runs = api.runs(project_name, filters={"group": "5-fold"})
    # 存储每个 fold 的最佳 metrics
    all_best_metrics = []

    for run in runs:
        print(f"------------------Processing run: {run.name} (ID: {run.id})----------------")
        # 获取该 run 的所有历史记录
        history = run.scan_history()
        # 找到 val_loss 最低的 epoch
        min_val_loss = float("inf")
        best_epoch_metrics = None
        for row in history:
            if "val/loss" in row and row["val/loss"] < min_val_loss:
                min_val_loss = row["val/loss"]
                best_epoch_metrics = row
        # 记录该 run 的最佳 metrics
        if best_epoch_metrics:
            print(f"Best epoch metrics for {run.name}:")
            for key, value in best_epoch_metrics.items():
                print(f"{key}: {value}")
            all_best_metrics.append(best_epoch_metrics)
        else:
            print(f"No metrics found for {run.name}.")

    # 计算所有 fold 的 metrics 的平均值
    if all_best_metrics:
        avg_metrics = {}
        for key in all_best_metrics[0].keys():
            avg_metrics[key] = np.mean([metrics[key] for metrics in all_best_metrics if key in metrics])
        print(
            "==============================Average metrics across all folds (lowest val/loss epoch)=====================")
        for key, value in avg_metrics.items():
            print(f"{key}: {value}")
    else:
        print("No metrics found for any run.")


if __name__ == "__main__":
    # metrics_from_wandb(project_name="audiossl_2025-01-31_12-55")
    # exit(0)
    # ori_test_tsv = "/20A021/ccomhuqin/meta1-1/eval/eval_rm_intersect.tsv"
    # ori_test_dur = "/20A021/ccomhuqin/meta1-1/eval/eval_durations.tsv"
    gt_test_tsv = "/20A021/finetune_music_dataset/exp/audiossl/1-1/eval_rm_intersect.tsv"
    gt_test_dur = "/20A021/finetune_music_dataset/exp/audiossl/1-1/eval_durations.tsv"
    # rename_gt_atst(ori_test_tsv, ori_test_dur, gt_test_tsv, gt_test_dur) #只跑一遍

    # ----------------- ATST-------------
    for k in range(5):
        print(f'Calculating metrics for fold_{k + 1}.....')
        atst_metrics_dir = f"/20A021/finetune_music_dataset/exp/audiossl/1-1/debug/fold_{k + 1}/metrics_test/"
        predictions_dir = os.path.join(atst_metrics_dir, 'predictions')
        generate_pred_tsv_atst(pred_path=predictions_dir, save_path=atst_metrics_dir)

        check_pred_match_gt(pred_csv=atst_metrics_dir + "pred_all.csv", gt_tsv=gt_test_tsv, gt_duration_tsv=gt_test_dur)
        compute_metrics(threshold=0, pred_csv=atst_metrics_dir + "pred_all.csv",
                        gt_tsv=gt_test_tsv, gt_dur=gt_test_dur,
                        save_path=atst_metrics_dir)
        compute_framewise_metrics(pred_csv=atst_metrics_dir + "pred_all.csv",
                                  gt_tsv=gt_test_tsv, dur_tsv=gt_test_dur, save_path=atst_metrics_dir)

    # ---------------MERT-------------------
    # mert_metrics_dir = '/20A021/compare_with/mert/1-1/freeze/0116-lr0.01/results/'
    # predictions_dir = os.path.join(mert_metrics_dir, 'predictions')
    # generate_pred_tsv_atst(pred_path=predictions_dir, save_path=mert_metrics_dir)
    #
    # check_pred_match_gt(pred_csv=mert_metrics_dir + "pred_all.csv", gt_tsv=gt_test_tsv, gt_duration_tsv=gt_test_dur)
    # # compute_metrics(threshold=0, pred_csv=mert_metrics_dir + "pred_all.csv",
    # #                 gt_tsv=gt_test_tsv, gt_dur=gt_test_dur,
    # #                 save_path=mert_metrics_dir)
    # compute_framewise_metrics(pred_csv=mert_metrics_dir + "pred_all.csv",
    #                           gt_tsv=gt_test_tsv, dur_tsv=gt_test_dur, save_path=mert_metrics_dir)
