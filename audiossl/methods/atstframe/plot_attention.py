import matplotlib.pyplot as plt
import os

import pandas as pd
import torch
import torchaudio
from sklearn.manifold import TSNE
import numpy as np

from audiossl.datasets.as_strong_utils.as_strong_dict import get_lab_dict
from model import FrameATSTLightningModule
from audiossl.transforms.common import MinMax
from pathlib import Path


def plot_spec(x, save_path):
    t = range(0, x.shape[0])
    f = range(0, x.shape[1])
    # plt.xlabel('frequency',fontsize=20)
    # plt.ylabel('time',fontsize=20)

    plt.axis('off')
    plt.pcolormesh(x)
    plt.xticks([])
    plt.yticks([])
    plt.margins(0, 0)
    # plt.xlabel('frequency',fontsize=20)
    # plt.ylabel('time',fontsize=20)

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_att(attentions, save_path, name):
    plot_spec(torch.sum(attentions[0, :, :, :], dim=0).cpu().numpy(), os.path.join(save_path, name + "att_headsum.png"))
    for i in range(attentions.shape[1]):
        plot_spec(attentions[0, i, :, :].cpu().numpy(), os.path.join(save_path, name + "att_head{}.png".format(i)))
        plt.imsave(fname=os.path.join(save_path, name + "att_head{}_imsave.png".format(i)),
                   arr=attentions[0, i].cpu().numpy(), dpi=300,
                   format='png')


def wav2mel(wav_file):
    melspec_t = torchaudio.transforms.MelSpectrogram(
        16000, f_min=60, f_max=7800, hop_length=160, win_length=1024, n_fft=1024, n_mels=64)
    to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)

    normalize = MinMax(min=-79.6482, max=50.6842)

    audio, sr = torchaudio.load(wav_file)
    assert sr == 16000
    melspec = normalize(to_db(melspec_t(audio)))
    return melspec


def get_pretrained_encoder(pretrained_ckpt_path):
    # get pretrained encoder
    print(f"Load pretrained ckpt {pretrained_ckpt_path}")
    s = torch.load(pretrained_ckpt_path)

    if 'pytorch-lightning_version' in s.keys():
        pretrained_model = FrameATSTLightningModule.load_from_checkpoint(
            pretrained_ckpt_path)
        pretrained_encoder = pretrained_model.model.teacher.encoder
    return pretrained_encoder


def mel2att(mel, model):
    return model.get_last_selfattention(mel.unsqueeze(0))


def plot_tsne_for_all_files(model, save_dir, save_name):
    tsv = "/20A021/ccomhuqin_seg/meta1-1/train/train.tsv"
    df = pd.read_csv(tsv, sep='\t')
    labels_df = df.groupby('filename')
    features_list, labels_list = [], []
    for name, group in labels_df:
        #basename = os.path.splitext(os.path.basename(name))[0]
        #ori_filename, start, end = basename.split('_')
        #if ori_filename.startswith('汾水情1') and int(start) % 10 == 0:
        print(name)
        mel = wav2mel(name).to(device)
        features, frame_labels = mel2features(mel, model, group)
        features_list.append(features)
        labels_list.append(frame_labels)
    all_features = np.vstack(features_list)  # 形状为 (250*360, 768)
    all_labels = np.concatenate(labels_list)  # 形状为 (250*360,)
    # 保存结果
    feature_save_path = os.path.join(save_dir, f"tsne_features_{save_name}.npy")
    label_save_path = os.path.join(save_dir, f"tsne_labels_{save_name}.npy")
    np.save(feature_save_path, all_features)
    np.save(label_save_path, all_labels)
    print(f"Save to {feature_save_path} and {label_save_path}")
    plot_tsne(save_dir, save_name, sampling_ratio=0.05)  # 每个类别采样5%


def plot_tsne(save_dir, save_name, sampling_ratio=0.01, min_sample_size=100):
    # 假设 features 是原始数据，形状为 (n_samples, n_features)
    all_features_tsne = np.load(os.path.join(save_dir, f"tsne_features_{save_name}.npy"), allow_pickle=True)
    all_labels = np.load(os.path.join(save_dir, f"tsne_labels_{save_name}.npy"), allow_pickle=True)
    # 获取所有唯一的类别
    unique_labels = np.unique(all_labels)
    # 初始化存储采样结果的列表
    sampled_features_list = []
    sampled_labels_list = []

    # 对每个类别进行采样
    for label in unique_labels:
        # 找到当前类别的所有样本
        indices = np.where(all_labels == label)[0]
        # 计算当前类别的采样数量
        n_samples = int(len(indices) * sampling_ratio)
        # 随机采样
        sampled_indices = np.random.choice(indices, n_samples, replace=False)
        # 保存采样结果
        sampled_features_list.append(all_features_tsne[sampled_indices])
        sampled_labels_list.append(all_labels[sampled_indices])
        print(f"Sample size of {label} is {n_samples}")

    # 合并采样结果
    sampled_features = np.vstack(sampled_features_list)
    sampled_labels = np.concatenate(sampled_labels_list)

    tsne = TSNE(n_components=2, random_state=42)
    features_tsne = tsne.fit_transform(sampled_features)

    # 获取所有唯一的标签
    unique_labels = np.unique(sampled_labels)
    # 为每个标签分配颜色
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
    color_map = {label: color for label, color in zip(unique_labels, colors)}

    # 绘制所有数据点，按标签设置颜色
    for label in unique_labels:
        indices = np.where(sampled_labels == label)[0]  # 找到当前标签对应的数据点索引
        plt.scatter(features_tsne[indices, 0],
                    features_tsne[indices, 1],
                    color=color_map[label], label=label, s=10)

    plt.title('t-SNE Visualization')
    plt.legend()
    plt.savefig(save_dir + f'tsne_sample_{save_name}_{sampling_ratio}.png', bbox_inches='tight')
    plt.close()

def mel2features(mel, model, label_df):
    # 得到frames的特征图
    model.eval()
    with torch.no_grad():
        features = model.get_frame_output(mel.unsqueeze(0))
    bs, n_frames, feature_dim = features.shape
    features = features.view(-1, feature_dim)  # 合并batch_size和frames成一个维度
    features_np = features.cpu().numpy()

    # 生成对应标签
    frame_duration = 0.04  # 40ms
    #frame_times = [i * frame_duration for i in range(n_frames)]  # 每帧对应的时间点
    frame_labels = np.full(n_frames, "NA", dtype=object)  # 默认值为 "no_label"
    for idx, row in label_df.iterrows():
        onset, offset, label = row["onset"], row["offset"], row["event_label"]
        for i in range(n_frames):
            frame_start = i * frame_duration
            frame_end = (i + 1) * frame_duration
            # 计算重叠时间
            overlap = min(offset, frame_end) - max(onset, frame_start)
            # 如果重叠时间大于帧的一半时间，则标记该帧
            if overlap > frame_duration / 2:
                frame_labels[i] = label

    return features_np, frame_labels


def vis_tsne_one_file(wav_file, mel, model, save_path, dim=2):
    # 得到frames的特征图
    model.eval()  # 将模型设置为评估模式
    with torch.no_grad():
        features = model.get_frame_output(mel.unsqueeze(0))
    bs, n_frames, feature_dim = features.shape
    features = features.view(-1, feature_dim)  # 合并batch_size和frames成一个维度
    features_np = features.cpu().numpy()
    tsne = TSNE(n_components=dim, random_state=42)
    features_tsne = tsne.fit_transform(features_np)

    # 生成对应标签
    frame_duration = 0.04  # 40ms
    frame_times = [i * frame_duration for i in range(n_frames)]  # 每帧对应的时间点
    frame_labels = np.full(n_frames, "NA", dtype=object)  # 默认值为 "no_label"
    tsv = "/20A021/ccomhuqin_seg/meta1-1/train/train.tsv"
    labels_df = pd.read_csv(tsv, sep='\t')
    labels_df = labels_df[labels_df["filename"] == wav_file]
    for idx, row in labels_df.iterrows():
        onset, offset, label = row["onset"], row["offset"], row["event_label"]
        for i in range(n_frames):
            frame_start = i * frame_duration
            frame_end = (i + 1) * frame_duration
            # 计算重叠时间
            overlap = min(offset, frame_end) - max(onset, frame_start)
            # 如果重叠时间大于帧的一半时间，则标记该帧
            if overlap > frame_duration / 2:
                frame_labels[i] = label
    unique_labels = np.unique(frame_labels)
    color_map = {label: color for label, color in zip(unique_labels, plt.cm.tab10.colors)}

    if dim == 3:
        # 创建 3D 图形
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        for label in unique_labels:
            indices = np.where(frame_labels == label)[0]  # 找到当前标签对应的帧索引
            ax.scatter(
                features_tsne[indices, 0],  # x 轴
                features_tsne[indices, 1],  # y 轴
                features_tsne[indices, 2],  # z 轴
                color=color_map[label],  # 颜色根据标签区分
                label=label,  # 标签
                #s=7,  # 点的大小
                alpha=0.8  # 点的透明度
            )
        # 设置标题和坐标轴标签
        ax.legend()
        ax.set_title('3D t-SNE Visualization')
    elif dim == 2:
        plt.figure(figsize=(10, 8))
        for label in unique_labels:
            indices = np.where(frame_labels == label)[0]  # 找到当前标签对应的帧索引
            plt.scatter(features_tsne[indices, 0], features_tsne[indices, 1], color=color_map[label], label=label)
        plt.title('2D t-SNE Visualization')
        plt.legend()
    plt.savefig(save_path + f'tsne_single_file_{dim}D.png', bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    wav_file = "/20A021/ccomhuqin_seg/data/train/低音板胡/串调1-1_0_10.wav"
    ckpt_path = "/20A021/dataset_from_dyl/save_path/pretrainBase-0916/last.ckpt"
    tsne_save_dir = "/20A021/dataset_from_dyl/save_path/pretrainBase-0916/tsne/"
    plot_tsne(tsne_save_dir, 'all', 0.02)
    exit(0)

    os.makedirs(tsne_save_dir, exist_ok=True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    mel = wav2mel(wav_file).to(device)
    # plot_spec(mel[0].cpu().numpy(),os.path.join(save_path,"mel.png"))

    model = get_pretrained_encoder(ckpt_path).to(device)
    #vis_tsne_one_file(wav_file, mel, model, save_path=tsne_save_dir, dim=3)
    plot_tsne_for_all_files(model, tsne_save_dir, 'all')

    # att = mel2att(mel, model)
    # print("len of attention: ", len(att))
    #
    # for i,att_ in enumerate(att):
    #     plot_att(att_, save_path, name="{}-".format(i))
