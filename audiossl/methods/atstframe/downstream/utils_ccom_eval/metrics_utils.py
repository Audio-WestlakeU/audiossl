from collections import defaultdict
import numpy as np
import sys
from scipy.stats import hmean
from mir_eval.util import midi_to_hz
from mir_eval.multipitch import evaluate as evaluate_frames
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

def notes_to_frames_no_inference(roll):
    time = np.arange(roll.shape[-1])
    freqs = [roll[:, t].nonzero()[0] for t in time]
    return time, freqs


def compute_mir_metrics(pred_inst, Yte, feature_rate):
    print(f"Compute using mir_eval, feature_rate is {feature_rate}")
    # copy-paste from Mertech
    metrics = defaultdict(list)
    eps = sys.float_info.epsilon
    t_ref, f_ref = notes_to_frames_no_inference(Yte)
    t_est, f_est = notes_to_frames_no_inference(pred_inst)

    t_ref = t_ref.astype(np.float64) / feature_rate
    f_ref = [np.array([midi_to_hz(21 + midi) for midi in freqs]) for freqs in f_ref]
    t_est = t_est.astype(np.float64) / feature_rate
    f_est = [np.array([midi_to_hz(21 + midi) for midi in freqs]) for freqs in f_est]

    IPT_frame_metrics = evaluate_frames(t_ref, f_ref, t_est, f_est)
    metrics['metric/IPT_frame/f1'].append(
        hmean([IPT_frame_metrics['Precision'] + eps, IPT_frame_metrics['Recall'] + eps]) - eps)

    for key, loss in IPT_frame_metrics.items():
        metrics['metric/IPT_frame/' + key.lower().replace(' ', '_')].append(loss)

    return metrics


def compute_sklearn_metrics(pred, target):
    print("-------------sklearn metrics----------")
    # 转换为 Scikit-learn 格式 [time_steps=2, num_labels=3]
    pred_sk = pred.T
    target_sk = target.T
    # Micro 指标，注意这里有个坑，在只有一个类别的情况下，用多类别的形式去计算，average要设置成"binary"，否则结果出错！
    micro_precision = precision_score(target_sk, pred_sk, average="micro")
    micro_recall = recall_score(target_sk, pred_sk, average="micro")
    micro_f1 = f1_score(target_sk, pred_sk, average="micro")

    # Macro 指标
    # macro_precision = precision_score(target_sk, pred_sk, average="macro", zero_division=0)
    # macro_recall = recall_score(target_sk, pred_sk, average="macro", zero_division=0)
    # macro_f1 = f1_score(target_sk, pred_sk, average="macro", zero_division=0)

    # 准确率
    # subset_accuracy = accuracy_score(target_sk, pred_sk)  # 严格匹配所有标签
    # flat_accuracy = accuracy_score(target_sk.flatten(), pred_sk.flatten())  # 等同于每个标签和时间步独立统计

    tp = np.sum((target_sk.flatten() == 1) & (pred_sk.flatten() == 1))
    fp = np.sum((target_sk.flatten() == 1) & (pred_sk.flatten() == 0))
    fn = np.sum((target_sk.flatten() == 0) & (pred_sk.flatten() == 1))


    print(f"Micro: precision[{micro_precision}], recall[{micro_recall}], f1[{micro_f1}]")
    # print(f"Macro: precision[{macro_precision}], recall[{macro_recall}], f1[{macro_f1}]")
    #print(f"Accuracy: [{subset_accuracy}], [{flat_accuracy}]")
    mir_accuracy = tp / (tp + fp + fn)
    #print(f"FP: {fp}, FN: {fn}")
    #print(f"手动计算mir_accuracy: {mir_accuracy}")

    return micro_precision, micro_recall, micro_f1, mir_accuracy
