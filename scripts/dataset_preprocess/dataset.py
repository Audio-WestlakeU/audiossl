from torch.utils import data
import torchaudio
import glob
import torch
import numpy as np
import os
from torch.nn import functional as F
import json
import pandas as pd
import time

import lmdb
import pickle
import tqdm
import pyarrow as pa

class Timer:
    def __init__(self):
        self.time=None
    def start(self):
        self.time = time.time()
    def release(self,name):
        print("{} time cost {}".format(name,time.time() - self.time))


class SpecDataset(data.Dataset):
    def __init__(self, path, ext="wav", sr=16000, max_len: float = 10, max_num: int = None):
        self.path = path
        self.audio_files = glob.glob(os.path.join(path, "*."+ext))
        self.sr = sr
        if max_num is not None and len(self.audio_files) > max_num:
            self.audio_files = self.audio_files[:max_num]
        self.max_len = max_len
        self.melspec_t = torchaudio.transforms.MelSpectrogram(
            sr, f_min=60, f_max=7800, hop_length=160, win_length=1024, n_fft=1024, n_mels=64)

    def __getitem__(self, index: int):
        waveform, sr = torchaudio.load(self.audio_files[index])
        if not (self.sr == sr):
            raise "sampling rate {} is expected, while {} is given".format(
                self.sr, sr)
        length = waveform.shape[-1]
        seg_len = int(sr)

        if length < seg_len:
            waveform = F.pad(waveform, (0, seg_len-length))
            length = seg_len

        start = np.random.randint(0, length-seg_len+1)
        seg1 = waveform[:, start:start+seg_len]
        start = np.random.randint(0, length-seg_len+1)
        seg2 = waveform[:, start:start+seg_len]

        logmelspec_seg1 = torch.log(self.melspec_t(seg1))
        logmelspec_seg2 = torch.log(self.melspec_t(seg2))

        return logmelspec_seg1, logmelspec_seg2

    def __len__(self):
        return len(self.audio_files)


class FSD50KDataset(data.Dataset):
    def __init__(self,  path, split="train", sr=16000, max_len: float = 10, max_num: int = None):
        self.path = path
        self.split = split
        labels_map = os.path.join(path, "lbl_map.json")
        with open(labels_map, 'r') as fd:
            self.labels_map = json.load(fd)
        self.labels_delim = ","
        self.sr = sr

        csv = None
        if split == "train":
            csv = "tr.csv"
        elif split == "valid":
            csv = "val.csv"
        elif split == "eval":
            csv = "eval.csv"
        else:
            raise "split shoud be one of train|valid|eval"

        df = pd.read_csv(os.path.join(path,csv))
        self.files = df['files'].values
        self.labels = df['labels'].values
        assert len(self.files) == len(self.labels)

        if max_num is not None and len(self.audio_files) > max_num:
            self.audio_files = self.audio_files[:max_num]
        self.max_len = max_len

        self.melspec_t = torchaudio.transforms.MelSpectrogram(
            sr, f_min=60, f_max=7800, hop_length=160, win_length=1024, n_fft=1024, n_mels=64 )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power",top_db=80)

    def _parse_labels(self, lbls: str) -> torch.Tensor:
        label_tensor = torch.zeros(len(self.labels_map)).float()
        for lbl in lbls.split(self.labels_delim):
            label_tensor[self.labels_map[lbl]] = 1

        return label_tensor

    def __getitem__(self, index: int):
        waveform, sr = torchaudio.load(self.files[index],normalize=True)
        if not (self.sr == sr):
            raise "sampling rate {} is expected, while {} is given".format(
                self.sr, sr)
        length = waveform.shape[-1]
        seg_len = int(sr)

        if length < seg_len:
            waveform = F.pad(waveform, (0, seg_len-length))
            length = seg_len

        if self.split == "train" or self.split == "valid":

            start = np.random.randint(0, length-seg_len+1)
            seg1 = waveform[:, start:start+seg_len]
            start = np.random.randint(0, length-seg_len+1)
            seg2 = waveform[:, start:start+seg_len]
            seg2 += 0.001 * torch.randn_like(seg2)

            if self.split == "train":
                start = np.random.randint(0, length-seg_len+1)
            else:
                start = (length - seg_len) // 2

            seg3 = waveform[:, start:start+seg_len]

            logmelspec_seg1 = self.norm_01(self.to_db(self.melspec_t(seg1)))
            logmelspec_seg2 = self.norm_01(self.to_db(self.melspec_t(seg2)))
            logmelspec_seg3 = self.norm_01(self.to_db(self.melspec_t(seg3)))

            label = self._parse_labels(self.labels[index])

            return (logmelspec_seg1, logmelspec_seg2, logmelspec_seg3), label

        else:

            logmelspec = self.to_db(self.melspec_t(waveform))
            label = self._parse_labels(self.labels[index])
            return logmelspec, label

    def norm_01(self,spec):
        var,mean = torch.var_mean(spec)
        spec -= mean
        spec/=torch.sqrt(var)
        return spec

    def __len__(self):
        return len(self.files)

class FSD50KDataset2(data.Dataset):
    def __init__(self,  path, split="train", sr=16000, max_len: float = 10, max_num: int = None, transform = None):
        self.path = path
        self.split = split
        labels_map = os.path.join(path, "lbl_map.json")
        with open(labels_map, 'r') as fd:
            self.labels_map = json.load(fd)
        self.labels_delim = ","
        self.num_classes = len(self.labels_map)

        self.sr = sr

        csv = None
        if split == "train":
            csv = "tr.csv"
        elif split == "valid":
            csv = "val.csv"
        elif split == "eval":
            csv = "eval.csv"
        else:
            raise "split shoud be one of train|valid|eval"

        df = pd.read_csv(os.path.join(path,csv))
        self.files = df['files'].values
        self.labels = df['labels'].values
        assert len(self.files) == len(self.labels)

        if max_num is not None and len(self.files) > max_num:
            self.files = self.files[:max_num]
        self.max_len = max_len
        self.timer = Timer()

        self.transform = transform


    def _parse_labels(self, lbls: str) -> torch.Tensor:
        label_tensor = torch.zeros(len(self.labels_map)).float()
        for lbl in lbls.split(self.labels_delim):
            label_tensor[self.labels_map[lbl]] = 1

        return label_tensor

    def __getitem__(self, index: int):
        file_path = self.files[index]
        waveform, sr = torchaudio.load(file_path,normalize=True)
        waveform = torchaudio.functional.resample(waveform,sr,self.sr)
        length = waveform.shape[-1]
        seg_len = int(self.sr)

        if length < seg_len:
            waveform = F.pad(waveform, (0, seg_len-length))
            length = seg_len

        
        label = self._parse_labels(self.labels[index])

        if self.transform is not None:
            waveform,label =  self.transform(waveform),label
        return waveform,label

    def norm_01(self,spec):
        var,mean = torch.var_mean(spec)
        spec -= mean
        spec/=torch.sqrt(var)
        return spec

    def __len__(self):
        return len(self.files)

class FSD50KDataset3(data.Dataset):
    def __init__(self,  path, split="train", multilabel=True, sr=16000, max_len: float = 10, max_num: int = None, transform = None):
        self.path = path
        self.split = split
        self.multilabel = multilabel
        labels_map = os.path.join(path, "lbl_map.json")
        with open(labels_map, 'r') as fd:
            self.labels_map = json.load(fd)
        self.labels_delim = ","
        self.num_classes = len(self.labels_map)

        self.sr = sr

        csv = None
        if split == "train":
            csv = "tr.csv"
        elif split == "valid":
            csv = "val.csv"
        elif split == "eval":
            csv = "eval.csv"
        else:
            raise "split shoud be one of train|valid|eval"

        df = pd.read_csv(os.path.join(path,csv))
        self.files = df['files'].values
        self.labels = df['labels'].values
        assert len(self.files) == len(self.labels)

        if max_num is not None and len(self.files) > max_num:
            self.files = self.files[:max_num]
        self.max_len = max_len

        self.transform = transform


    def _parse_labels(self, lbls: str) -> torch.Tensor:
        if self.multilabel:
            label_tensor = torch.zeros(len(self.labels_map)).float()
            for lbl in lbls.split(self.labels_delim):
                label_tensor[self.labels_map[lbl]] = 1
        else:
            assert len(lbls.split(self.labels_delim)) == 1
            label_tensor = torch.tensor(self.labels_map[lbls],dtype=torch.int16)
        return label_tensor

    def __getitem__(self, index: int):
        file_path = self.files[index]
        waveform, sr = torchaudio.load(file_path,normalize=True)
        waveform = torchaudio.functional.resample(waveform,sr,self.sr)
        length = waveform.shape[-1]
        seg_len = int(self.sr)
        
        label = self._parse_labels(self.labels[index])

        if self.transform is not None:
            waveform,label =  self.transform(waveform),label
        return waveform,label,os.path.basename(file_path)

    def norm_01(self,spec):
        var,mean = torch.var_mean(spec)
        spec -= mean
        spec/=torch.sqrt(var)
        return spec

    def __len__(self):
        return len(self.files)


class LMDBDataset(data.Dataset):
    def __init__(self, db_path,split, transform=None, target_transform=None):
        self.db_path = db_path
        lmdb_path = None
        if split=="train":
            lmdb_path = os.path.join(self.db_path,"train.lmdb")
        elif split=="valid":
            lmdb_path = os.path.join(self.db_path,"valid.lmdb")
        else:
            lmdb_path = os.path.join(self.db_path,"eval.lmdb")
        self.env = lmdb.open(lmdb_path, subdir=os.path.isdir(lmdb_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            # self.length = txn.stat()['entries'] - 1
            self.length =pa.deserialize(txn.get(b'__len__'))
            self.keys= pa.deserialize(txn.get(b'__keys__'))

        self.transform = transform
        self.target_transform = target_transform
        with self.env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[0])
        unpacked = pa.deserialize(byteflow)
        self.num_classes = unpacked[1].shape[1]
        self.txn = self.env.begin(write=False)

        self.sr = 16000
        self.timer = Timer()

    def __getitem__(self, index):
        env = self.env
        start_time = time.time()

        byteflow = self.txn.get(self.keys[index])
        get_time = time.time() - start_time
        unpacked = pa.deserialize(byteflow)

        waveform, label = torch.from_numpy(unpacked[0]).squeeze(0),torch.from_numpy(unpacked[1]).squeeze(0)
        length = waveform.shape[-1]
        seg_len = int(self.sr)

        if length < seg_len:
            waveform = F.pad(waveform, (0, seg_len-length))
            length = seg_len

        if self.transform is not None:
            transform_start = time.time()
            waveform = self.transform(waveform)
            transform_time = time.time() - transform_start
        

        return waveform, label 

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'


