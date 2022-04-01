import lmdb
import pyarrow as pa
from torch.nn import Module
import torch.utils.data as data
import os
import torch
import random
random.seed(1234)

from copy import deepcopy

class LMDBDataset(data.Dataset):
    def __init__(self, db_path,split, subset=None, transform=None, target_transform=None):
        self.db_path = db_path
        lmdb_path = None
        if split=="train":
            lmdb_path = os.path.join(self.db_path,"train.lmdb")
        elif split=="valid":
            lmdb_path = os.path.join(self.db_path,"valid.lmdb")
        else:
            lmdb_path = os.path.join(self.db_path,"eval.lmdb")
        self.subset = subset
        self.env = lmdb.open(lmdb_path, subdir=os.path.isdir(lmdb_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            # self.length = txn.stat()['entries'] - 1
            
            self.length =pa.deserialize(txn.get(b'__len__'))
            self.keys= pa.deserialize(txn.get(b'__keys__'))
            self.org_keys = deepcopy(self.keys)
            self.start = 0
            if subset is not None and subset < self.length:
                self.length = subset
                random.shuffle(self.keys)
                self.org_keys = deepcopy(self.keys)
                self.keys = self.keys[:subset]
                self.start=subset

        self.transform = transform
        self.target_transform = target_transform
        with self.env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[0])
        unpacked = pa.deserialize(byteflow)
        self.num_classes = unpacked[1].shape[1]
        self.txn = self.env.begin(write=False)

        self.sr = 16000

    def __getitem__(self, index):
        env = self.env

        byteflow = self.txn.get(self.keys[index])
        unpacked = pa.deserialize(byteflow)

        waveform, label = torch.from_numpy(unpacked[0]).squeeze(0),torch.from_numpy(unpacked[1]).squeeze(0)
        length = waveform.shape[-1]
        if length > self.sr * 5:
            length = 501 
        else:
            length = length // 160 + 1

        seg_len = int(self.sr)


        if self.transform is not None:
            transformed = self.transform(waveform)
            if self.target_transform is not None:
                transformed = list(transformed)
                transformed[0],label = self.target_transform(transformed[0],label)
                transformed = tuple(transformed)
        
            return transformed, label 
        else:
            return waveform, label
    def cycle(self):
        if self.start + self.subset > len(self.org_keys):
            self.keys = self.org_keys[self.start:] + self.org_keys[:self.start+self.subset - len(self.org_keys)]
            random.shuffle(self.org_keys)
            self.start = 0
            #self.start = self.start + self.subset - len(self.org_keys)
        else:
            self.keys = self.org_keys[self.start:self.start+self.subset]
            self.start = self.start+self.subset

    def __len__(self):
        return len(self.keys)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'
