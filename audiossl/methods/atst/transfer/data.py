from pytorch_lightning import LightningDataModule
from torch.utils import data
from audiossl.methods.atst.transfer.utils import load_dataset
from audiossl.methods.atst.transfer.transform import FreezingTransferTrainTransform
import torch
import torch.nn.functional as F
from audiossl import datasets


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

class FreezingTransferDataModule(LightningDataModule):
    def __init__(self,
                 data_path:str,
                 dataset_name:str,

                 fold:int = 0,
                 batch_size_per_gpu=1024,
                 num_workers=10,
                 **kwargs
                 ):
        super().__init__()
        self.batch_size=min(512,batch_size_per_gpu)
        self.num_workers=num_workers
        dataset_train,dataset_val,dataset_test = load_dataset(dataset_name,
                                                              data_path,
                                                              fold,
                                                              FreezingTransferTrainTransform())
        transform = FreezingTransferTrainTransform()
        dataset_info=datasets.get_dataset(dataset_name)
        num_folds = dataset_info.num_folds
        self.num_labels=dataset_info.num_labels
        self.multi_label=dataset_info.multi_label
        if num_folds > 1:
            self.dataset_train = dataset_info.creator(data_path,
                                                      "train",
                                                      fold,
                                                      transform,
                                                      target_transform=None)
            self.dataset_val = dataset_info.creator(data_path,
                                                      "valid",
                                                      fold,
                                                      transform,
                                                      target_transform=None)
            self.dataset_test = dataset_info.creator(data_path,
                                                      "test",
                                                      fold,
                                                      transform,
                                                      target_transform=None)
        else:
            self.dataset_train = dataset_info.creator(data_path,
                                                      "train",
                                                      transform,
                                                      target_transform=None)
            self.dataset_val = dataset_info.creator(data_path,
                                                      "valid",
                                                      transform,
                                                      target_transform=None)
            self.dataset_test = dataset_info.creator(data_path,
                                                      "test",
                                                      transform,
                                                      target_transform=None)
    def prepare_data(self):
        pass

    def train_dataloader(self):
        return data.DataLoader(self.dataset_train,
                        batch_size=self.batch_size,
                        num_workers=self.num_workers,
                        shuffle=False,
                        sampler=None,
                        drop_last=False,
                        collate_fn=collate_fn,
                        pin_memory=True)
    def val_dataloader(self):
        return data.DataLoader(self.dataset_val,
                        batch_size=self.batch_size,
                        num_workers=self.num_workers,
                        shuffle=False,
                        sampler=None,
                        drop_last=False,
                        collate_fn=collate_fn,
                        pin_memory=True)
    def test_dataloader(self):
        return data.DataLoader(self.dataset_test,
                        batch_size=self.batch_size,
                        num_workers=self.num_workers,
                        shuffle=False,
                        sampler=None,
                        drop_last=False,
                        collate_fn=collate_fn,
                        pin_memory=True)


    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("FreezingTransferData")
        parser.add_argument("--dataset_name", type=str, default=None, help="dataset name")
        parser.add_argument("--data_path", type=str, default=None, help="data path")
        parser.add_argument("--fold", type=int, default=0, help="fold, only used in datasets containing N folds")
        parser.add_argument('--batch_size_per_gpu', default=256, type=int,
            help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
        parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
        return parent_parser
