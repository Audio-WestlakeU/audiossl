import os.path
import pickle

import librosa
import torch
import numpy as np

from audiossl.datasets.as_strong_utils.as_strong_dict import get_lab_dict
from audiossl.datasets.dcase_utils import ManyHotEncoder
from audiossl.methods.atstframe.downstream.utils_psds_eval.gpu_decode import (
    onehot_decode_merge_preds,
    MedianPool2d
)

########### TODO: 将下面参数改成从frame_40.yaml中读取
SR = 16000
WINDOW_SIZE = 10
STRIDE = 5
HOP_LEN = 160


def write_results(filenames, predictions, save_dir):
    '''
    copy from hubert/src/my_train.py write_results
    filenames: a list of all filenames with full path
    predictions: concatenation of [bsz, cls, T]
    '''
    # merge results to original songs and write as independent files
    merge_preds = torch.cat(predictions, dim=0)  # 在第一个batch的维度拼接起来
    sample_size, C, T = merge_preds.shape
    assert sample_size == len(filenames)
    logits = merge_preds.transpose(1, 2)  # 变成[sample_size, T, C]，
    probs = torch.softmax(logits, dim=-1)
    file_params = {'file': [], 'start': [], 'end': []}

    for filename in filenames:
        file_name_no_suffix = os.path.splitext(os.path.basename(filename))[0]
        name_parts = file_name_no_suffix.split('_')
        start_time = int(name_parts[-2])
        end_time = int(name_parts[-1])
        source_file_name_no_suffix = '_'.join(name_parts[:-2])
        file_params['file'].append(source_file_name_no_suffix)
        file_params['start'].append(start_time)
        file_params['end'].append(end_time)

    results_list = [
        {
            'audio_id': file,
            'start': start,
            'end': end,
            'probs': probs
        } for (file, start, end, probs) in zip(
            file_params['file'], file_params['start'], file_params['end'], probs
        )
    ]
    results = dict()
    for item in results_list:
        # 把每一首的所有片段放在同一个id下，形成一个list
        results[item['audio_id']] = results.get(item['audio_id'], list()) + [item]

    with open(os.path.join(save_dir, 'results.pickle'), 'wb') as handle:
        pickle.dump(results, handle)

    # decode_results(save_dir, labels_list)


def decode_results(save_dir, pred_decoder, median_filter):
    with open(os.path.join(save_dir, 'results.pickle'), 'rb') as handle:
        results = pickle.load(handle)

    save_pred_dir = os.path.join(save_dir, 'predictions')
    os.makedirs(save_pred_dir, exist_ok=True)
    for filename in results:
        probs = collect_and_avg(results[filename], pred_decoder)
        dummy_threshold = 0  # 使用源代码的逻辑，这里的key是threshold，传入的是[0]
        decoded_strong = onehot_decode_merge_preds(
            probs,
            thresholds=[dummy_threshold],
            filenames=[filename],
            encoder=pred_decoder,
            median_filter=median_filter
        )
        df = decoded_strong[dummy_threshold].sort_values('onset')
        df_cleaned = df[df['event_label'] != 'NA']
        df_cleaned.to_csv(os.path.join(save_pred_dir, filename + '.csv'), index=False)

#
# def decode_k_fold_results(save_dirs, pred_decoder, median_filter):
#


def collect_and_avg(result_dicts, pred_decoder: ManyHotEncoder):
    '''
    crop out results to window_size
    stack and avg results wrt stride
    '''
    time_ranges = [[item['start'], item['end']] for item in result_dicts]
    # idx_ranges = [[int(pred_decoder._time_to_frame(value)) for value in row] for row in time_ranges]

    # 1. 确定全局时间范围
    global_start = min(start for start, end in time_ranges)
    global_end = max(end for start, end in time_ranges)
    global_length = pred_decoder.time_to_frame(global_end - global_start)

    # 2. 初始化结果张量和计数张量
    result = torch.zeros(global_length, NUM_LABELS)
    count = torch.zeros(global_length, dtype=torch.int32)

    # 3. 累加张量并记录计数
    for i, result_dict in enumerate(sorted(result_dicts, key=lambda x: x['start'])):
        start_idx = pred_decoder.time_to_frame(result_dict['start'] - global_start)
        end_idx = pred_decoder.time_to_frame(result_dict['end'] - global_start)
        result[start_idx:end_idx] += result_dict['probs'].cpu()
        count[start_idx:end_idx] += 1

    # 4. 计算平均值，避免除以零
    count = count.unsqueeze(1)  # 形状扩展为 [global_length, 1] 以便广播
    result = result / count.clamp(min=1)  # 防止被零除

    return result


def plain_decode(probs):
    '''
    probs -> label
    without any post-processing decoding technique
    also: auto aggregate
    '''
    label = probs.argmax(-1).numpy()
    results = [[label[0], 0, 1]]
    for i, l in enumerate(label[1:], 1):
        prevl, prev_start, prev_end = results[-1]
        if prevl == l and prev_end == i:
            # same label, next to each other
            results[-1][-1] = i + 1
        else:
            # new event
            results.append([l, i, i + 1])
    return results


def interprete(labels, labels_list):
    '''
    label -> text
    $labelname,$start,$end
    '''
    texts = list()
    for (label_idx, start, end) in labels:
        start, end = map(lambda x: librosa.frames_to_time(x, sr=SR, hop_length=320), [start, end])
        texts.append([labels_list[label_idx], start, end])
    return texts


if __name__ == "__main__":
    labels_list = list(get_lab_dict("/20A021/ccomhuqin_seg/meta1-1/common_labels_na.txt").keys())
    NUM_LABELS = len(labels_list)
    pred_decoder = ManyHotEncoder(
        labels_list,
        audio_len=10,  # self.config["data"]["audio_max_len"],
        frame_len=1024,  # self.config["feats"]["n_filters"],
        frame_hop=160,  # self.config["feats"]["hop_length"],
        net_pooling=4,  # self.config["data"]["net_subsample"],
        fs=16000,  # self.config["data"]["fs"],
    )
    median_filter = MedianPool2d(7, same=True)  # freeze_mode下为了对比不同的模型，不用filter
    for k in range(5):
        print(f'Decoding fold_{k + 1}...')
        decode_results(f"/20A021/finetune_music_dataset/exp/audiossl/1-1/debug/fold_{k + 1}/metrics_test/",
                       pred_decoder, median_filter=median_filter)
