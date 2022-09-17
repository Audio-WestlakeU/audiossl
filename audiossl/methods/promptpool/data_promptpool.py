from pytorch_lightning import LightningDataModule
from torch.utils import data
from audiossl.datasets import LMDBDataset
from transform_cls import ClsPromptTrainTransform
import torch
import pyarrow as pa

class PromptPoolDataset(LMDBDataset):
    def __init__(self, db_path,split, queries , subset=None, transform=None, target_transform=None):
        super().__init__(db_path,split,subset=subset,transform=transform)
        self.queries = queries
        print(len(self.keys))
        print(len(queries.keys()))
        self.keys = list(queries.keys())
    
    def __getitem__(self, index):
        env = self.env

        key = self.keys[index]
        query = self.queries[key]
        byteflow = self.txn.get(key)
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
        
            return transformed, [query]*2, label 
        else:
            return waveform, query, label

class PromptPoolDataModule(LightningDataModule):
    def __init__(self,
                 queries,
                 data_path=None,
                 transform=None,
                 batch_size_per_gpu=256,
                 num_workers=10,
                 subset=200000,
                 **kwargs,
                 ):
        super().__init__()
        self.dataset=PromptPoolDataset(data_path,
                                 queries=queries,
                                 split="train",
                                 subset=subset,
                                 transform=transform)
        self.batch_size=batch_size_per_gpu
        self.num_workers=num_workers
        self.save_hyperparameters(ignore=['queries','transform'])
    

    def train_dataloader(self):

        return data.DataLoader(self.dataset,
                               batch_size=self.batch_size,
                               num_workers=self.num_workers,
                               sampler=None,
                               drop_last=True)

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("ATSTData")
        parser.add_argument("--data_path", type=str, default=None, help="data path")
        parser.add_argument('--batch_size_per_gpu', default=256, type=int,
            help='Per-GPU batch-size : number of distinct samples loaded on one GPU.')
        parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
        parser.add_argument('--subset', default=200000, type=int, help='subset of training data')
        return parent_parser
