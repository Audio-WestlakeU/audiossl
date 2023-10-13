from audiossl import datasets
import torch
import pyarrow as pa
import numpy as np
from torch.utils.data import WeightedRandomSampler
from audiossl.lightning.utils import EmbeddingExtractor
from torch.utils.data import DataLoader

from pytorch_lightning import LightningModule
import sys

class Lable(LightningModule):
    def __init__(self):
        super().__init__()
    def forward(self,batch):
        return batch[1],batch[2]

class Transform:
    def __call__(self,x):
        return torch.tensor(0)


def get_label(path,index):
    lmdb_path = path + "/train.lmdb"
    env = lmdb.open(lmdb_path, subdir=os.path.isdir(lmdb_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
    txn = env.begin(write=False)
    byteflow = txn.get(d.keys[index])
    env.close()
    return label

def get_labels(d):
    labels = np.array([0]*len(d))
    for index in range(len(d)):
        byteflow=d.txn.get(d.keys[index])
        unpacked = pa.deserialize(byteflow)

        label = unpacked[1].squeeze(0)
        label = np.argmax(label)
        labels[index] = label
        if index % 1000 == 0:
            print(index)

    labels = np.array(labels)
    idxs , counts = np.unique(labels, return_counts=True)
    weights_=1/counts

    weights = {idx:weights_[i] for i,idx in enumerate(idxs)}

    weights_labels = [weights[i] for i in labels]
    np.save("weights_labels.npy",np.array(weights_labels))
    sampler = WeightedRandomSampler(weights_labels, len(weights_labels))
    return sampler

path = sys.argv[1]


from torch.utils.data import ConcatDataset 
dub = datasets.LMDBDataset(path, "train", subset=None, transform=Transform(),return_key=True)
db = datasets.LMDBDataset(path + "../audioset_b", "train", subset=None, transform=Transform(),return_key=True)
d = ConcatDataset([dub,db])
#sampler=get_labels(d)
extractor = EmbeddingExtractor(Lable(),1)
result = extractor.extract(DataLoader(d,batch_size=512,num_workers=20,shuffle=False))
result = [r for r in zip(*result)]

labels,keys=torch.cat(result[0],dim=0),list(sum(result[1],()))

counts = torch.sum(labels,dim=0)
weights_labels = torch.sum(labels*1000/(torch.sum(labels,dim=0)+0.01),dim=-1)

save_dict = {"keys":keys,"weights_labels":weights_labels}
torch.save(save_dict,"weights_labels.pt5")
