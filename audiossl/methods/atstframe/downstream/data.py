from pytorch_lightning import LightningDataModule
from torch.utils import data
import torch
import torch.nn.functional as F


def collate_fn(data):
    spec_l = []
    length_l = []
    label_l = []
    for d in data:
        spec_l.append(d[0][0])
        length_l.append(d[0][1])
        label_l.append(d[1])

    max_len = max(length_l)
    for i in range(len(spec_l)):
        spec_l[i]=F.pad(spec_l[i],(0,max_len-length_l[i]))
        length_l[i]=torch.tensor(length_l[i])
        label_l[i] = torch.tensor(label_l[i])
    return (torch.stack(spec_l),torch.stack(length_l)),torch.stack(label_l)
