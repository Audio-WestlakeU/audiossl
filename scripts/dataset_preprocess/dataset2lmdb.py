
import os
import os.path as osp
import os, sys
import os.path as osp

import lmdb
import tqdm
import pyarrow as pa

import torch.utils.data as data
from torch.utils.data import DataLoader
import dataset
import numpy as np

def dumps_pyarrow(obj):
    """
    Serialize an object.

    Returns:
        Implementation-dependent bytes-like object
    """
    return pa.serialize(obj).to_buffer()

def dataset2lmdb(dataset, save_prefix, write_frequency=5000, max_num=400000, num_workers=16):

    dataloader = DataLoader(dataset,num_workers=num_workers,shuffle=True)


    if len(dataset) > max_num:
        lmdb_split = 0
        lmdb_path = "{}_{}.lmdb".format(save_prefix,lmdb_split)
        lmdb_split += 1
    else:
        lmdb_path = "{}.lmdb".format(save_prefix)

    if os.path.exists(lmdb_path):
        print("{} already exists".format(lmdb_path))
        exit(0)
    isdir = os.path.isdir(lmdb_path)

    db = lmdb.open(lmdb_path, subdir=isdir,
                   map_size=1099511627776 // 4, readonly=False,
                   meminit=False, map_async=True)

    txn = db.begin(write=True)
    keys = []
    for idx, data in enumerate(dataloader):
        image, label, name = data[0].numpy(),data[1].numpy(),data[2][0]
        keys.append(u'{}'.format(name).encode('ascii'))
        txn.put(u'{}'.format(name).encode('ascii'), dumps_pyarrow((image, label)))
        if idx >0 and idx % max_num ==0:

            txn.commit()
            with db.begin(write=True) as txn:
                txn.put(b'__keys__', dumps_pyarrow(keys))
                txn.put(b'__len__', dumps_pyarrow(len(keys)))
            print("Flushing database to {} ...".format(lmdb_path))
            db.sync()
            db.close()

            lmdb_path = "{}_{}.lmdb".format(save_prefix,lmdb_split)
            lmdb_split += 1
            isdir = os.path.isdir(lmdb_path)

            db = lmdb.open(lmdb_path, subdir=isdir,
                        map_size=1099511627776 // 18, readonly=False,
                        meminit=False, map_async=True)

            txn = db.begin(write=True)
            keys = []

        if idx % write_frequency == 0:
            print("[%d/%d]" % (idx, len(dataloader)))
            txn.commit()
            txn = db.begin(write=True)
    txn.commit()
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', dumps_pyarrow(keys))
        txn.put(b'__len__', dumps_pyarrow(len(keys)))

    print("Flushing database ...")
    db.sync()
    db.close()

def folder2lmdb(dpath, split="train",lmdb_path=None, write_frequency=5000, max_num=200000, num_workers=5):

    if lmdb_path is None:
        lmdb_path=dpath
    ds = dataset.FSD50KDataset3(dpath,split=split)
    dataloader = DataLoader(ds,num_workers=num_workers,shuffle=True)
    i = iter(ds)


    if len(ds) > max_num:
        lmdb_split = 0
        lmdb_path = osp.join(lmdb_path, "{}_{}.lmdb".format(split,lmdb_split))
        lmdb_split += 1
    else:
        lmdb_path = osp.join(lmdb_path, "{}.lmdb".format(split))

    if os.path.exists(lmdb_path):
        print("{} already exists".format(lmdb_path))
        exit(0)
    isdir = os.path.isdir(lmdb_path)

    db = lmdb.open(lmdb_path, subdir=isdir,
                   map_size=1099511627776 // 4, readonly=False,
                   meminit=False, map_async=True)

    txn = db.begin(write=True)
    keys = []
    for idx, data in enumerate(ds):
        image, label, name = data[0].unsqueeze(0).numpy(),data[1].unsqueeze(0).numpy(),data[2]
        keys.append(u'{}'.format(name).encode('ascii'))
        txn.put(u'{}'.format(name).encode('ascii'), dumps_pyarrow((image, label)))
        if idx >0 and idx % max_num ==0:

            txn.commit()
            with db.begin(write=True) as txn:
                txn.put(b'__keys__', dumps_pyarrow(keys))
                txn.put(b'__len__', dumps_pyarrow(len(keys)))
            print("Flushing database to {} ...".format(lmdb_path))
            db.sync()
            db.close()

            lmdb_path = osp.join(lmdb_path, "{}_{}.lmdb".format(split,lmdb_split))
            lmdb_split += 1
            isdir = os.path.isdir(lmdb_path)

            db = lmdb.open(lmdb_path, subdir=isdir,
                        map_size=1099511627776 // 18, readonly=False,
                        meminit=False, map_async=True)

            txn = db.begin(write=True)
            keys = []

        if idx % write_frequency == 0:
            print("[%d/%d]" % (idx, len(ds)))
            txn.commit()
            txn = db.begin(write=True)
    txn.commit()
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', dumps_pyarrow(keys))
        txn.put(b'__len__', dumps_pyarrow(len(keys)))

    print("Flushing database ...")
    db.sync()
    db.close()

    return ds

class ImageFolderLMDB(data.Dataset):
    def __init__(self, db_path, transform=None, target_transform=None):
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=osp.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            # self.length = txn.stat()['entries'] - 1
            self.length =pa.deserialize(txn.get(b'__len__'))
            self.keys= pa.deserialize(txn.get(b'__keys__'))

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, target = None, None
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        unpacked = pa.deserialize(byteflow)

        waveform, label = unpacked
        if self.transform is not None:
            waveform = self.transform(waveform)


        return waveform, label

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'


if __name__ == "__main__":
    import sys
    path,split = sys.argv[1:]
    #folder2lmdb(path,split=split)
    #ds = dataset.FSD50KDataset3(path,split=split,multilabel=True)
    #dataset2lmdb(ds,save_prefix=os.path.join(path,split))
    folder2lmdb(path,split="train")

    """
    folder2lmdb(path,split="eval")
    folder2lmdb(path,split="train")
    """
    ds = ImageFolderLMDB(os.path.join(path,"train.lmdb"))
    i = iter(ds)
    next(i)
