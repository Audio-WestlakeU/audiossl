import json
import os

import torch
import torch.utils.data as data
import torchaudio
import pandas as pd

class MusicAudioDataset(data.Dataset):
    def __init__(self, manifest_path, split="train", transform=None, target_transform=None, multilabel=True,
                 return_key=False, subset=None):
        self.path = manifest_path
        self.split = split
        self.multilabel = multilabel
        labels_map = os.path.join(manifest_path, "lbl_map.json")
        with open(labels_map, 'r') as fd:
            self.labels_map = json.load(fd)
        self.labels_delim = ","
        self.num_classes = len(self.labels_map)

        if split == "train":
            tsv = "tr.tsv"
        elif split == "valid":
            tsv = "val.tsv"
        elif split == "eval":
            tsv = "eval.tsv"
        else:
            raise "split shoud be one of train|valid|eval"

        df = pd.read_csv(os.path.join(manifest_path, tsv), sep='\t')
        self.length = df.shape[0]
        if subset is not None and subset < self.length:
            df = df.sample(n=subset).reset_index(drop=True)
            self.length = subset
        self.files = df['files'].values
        self.labels = df['labels'].values
        self.transform = transform
        self.target_transform = target_transform
        self.return_key = return_key

    def _parse_labels(self, lbls: str) -> torch.Tensor:
        if self.multilabel:
            label_tensor = torch.zeros(len(self.labels_map)).float()
            for lbl in lbls.split(self.labels_delim):
                label_tensor[self.labels_map[lbl]] = 1
        else:
            assert len(lbls.split(self.labels_delim)) == 1
            label_tensor = torch.tensor(self.labels_map[lbls], dtype=torch.int16)
        return label_tensor

    def __getitem__(self, index: int):
        file_path = self.files[index]
        key = u'{}'.format(os.path.basename(file_path)).encode('ascii')
        waveform, sr = torchaudio.load(file_path, normalize=True)
        waveform_mono = torch.mean(waveform, dim=0, keepdim=True)
        label = self._parse_labels(self.labels[index])

        if self.transform is not None:
            transformed = self.transform(waveform_mono)
            if self.target_transform is not None:
                transformed = list(transformed)
                transformed[0], label = self.target_transform(transformed[0], label)
                transformed = tuple(transformed)

            if self.return_key:
                return transformed, label, key
            else:
                return transformed, label
        else:
            if self.return_key:
                return waveform, label, key
            else:
                return waveform, label

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.path + ')'