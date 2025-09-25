from pytorch_lightning import LightningDataModule
import torch
from torch.utils import data
from audiossl import datasets
import numpy as np
import torch.distributed as dist
import os
from torch.utils.data import ConcatDataset

def get_inmemory_datamodule(x_train,
                            y_train,
                            x_val,
                            y_val,
                            x_test,
                            y_test,
                            batch_size):
            
    dataset_train = data.TensorDataset(x_train,y_train)
    dataset_val = data.TensorDataset(x_val,y_val)
    dataset_test = data.TensorDataset(x_test,y_test)

    return LightningDataModule.from_datasets(dataset_train,dataset_val,dataset_test,batch_size=batch_size)
import numpy as np
import pyarrow as pa
from torch.utils.data import WeightedRandomSampler
from typing import Any, Iterable, Iterator, List, Optional, Sized, Tuple, Union
from torch.utils.data import Dataset, DistributedSampler, Sampler
class _DatasetSamplerWrapper(Dataset):
    """Dataset to create indexes from `Sampler` or `Iterable`"""

    def __init__(self, sampler: Union[Sampler, Iterable]) -> None:
        if not isinstance(sampler, Sized):
            raise TypeError(
                "You seem to have configured a sampler in your DataLoader which"
                " does not provide `__len__` method. The sampler was about to be"
                " replaced by `DistributedSamplerWrapper` since `use_distributed_sampler`"
                " is True and you are using distributed training. Either provide `__len__`"
                " method in your sampler, remove it from DataLoader or set `use_distributed_sampler=False`"
                " if you want to handle distributed sampling yourself."
            )
        if len(sampler) == float("inf"):
            raise TypeError(
                "You seem to have configured a sampler in your DataLoader which"
                " does not provide finite `__len__` method. The sampler was about to be"
                " replaced by `DistributedSamplerWrapper` since `use_distributed_sampler`"
                " is True and you are using distributed training. Either provide `__len__`"
                " method in your sampler which returns a finite number, remove it from DataLoader"
                " or set `use_distributed_sampler=False` if you want to handle distributed sampling yourself."
            )
        self._sampler = sampler
        # defer materializing an iterator until it is necessary
        self._sampler_list: Optional[List[Any]] = None

    def __getitem__(self, index: int) -> Any:
        if self._sampler_list is None:
            self._sampler_list = list(self._sampler)
        return self._sampler_list[index]

    def __len__(self) -> int:
        return len(self._sampler)

    def reset(self) -> None:
        """Reset the sampler list in order to get new sampling."""
        self._sampler_list = list(self._sampler)


#    class DistributedSamplerWrapper(DistributedSampler):
#    """Wrapper over ``Sampler`` for distributed training.
#    Allows you to use any sampler in distributed mode. It will be automatically used by Lightning in distributed mode if
#    sampler replacement is enabled.
#    Note:
#        The purpose of this wrapper is to take care of sharding the sampler indices. It is up to the underlying
#        sampler to handle randomness and shuffling. The ``shuffle`` and ``seed`` arguments on this wrapper won't
#        have any effect.
#    """
#
#    def __init__(self, sampler: Union[Sampler, Iterable], *args: Any, **kwargs: Any) -> None:
#        super().__init__(_DatasetSamplerWrapper(sampler), *args, **kwargs)
#
#    def __iter__(self) -> Iterator:
#        self.dataset.reset()
#        return (self.dataset[index] for index in super().__iter__())
class DistributedSamplerWrapper(DistributedSampler):
    def __init__(
            self, sampler, dataset,
            num_replicas=None,
            rank=None,
            shuffle: bool = True):
        super(DistributedSamplerWrapper, self).__init__(
            dataset, num_replicas, rank, shuffle)
        # source: @awaelchli https://github.com/PyTorchLightning/pytorch-lightning/issues/3238
        self.sampler = sampler

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        self.sampler.generator = g
        indices = list(self.sampler)
        self.sampler.generator = None # compatible for saving
        if self.epoch == 0:
            print(f"\n DistributedSamplerWrapper :  {indices[:10]} \n\n")
        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)

def get_sampler(d):
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
    sampler = WeightedRandomSampler(weights_labels, len(weights_labels))
    return sampler
# Add a batch sampler for DCASE concat dataset [MARK]
class DownstreamDataModule(LightningDataModule):
    def __init__(self,
                 data_path:str,
                 dataset_name:str,
                 fold:int = 0,
                 batch_size_per_gpu=1024,
                 num_workers=10,
                 transforms=[None,None,None],
                 target_transforms=[None,None,None],
                 collate_fn=None,
                 limit_batch_size=None,
                 shuffle = True,
                 sampler = None,
                 **kwargs
                 ):
        super().__init__()
        self.batch_size=min(limit_batch_size,batch_size_per_gpu)\
            if limit_batch_size is not None else batch_size_per_gpu
        self.num_workers=num_workers
        self.dataset_name = dataset_name
        self.transforms=transforms
        self.target_transforms=target_transforms
        self.collate_fn = collate_fn
        dataset_info=datasets.get_dataset(dataset_name)
        num_folds = dataset_info.num_folds
        self.num_labels=dataset_info.num_labels
        self.multi_label=dataset_info.multi_label
        self.shuffle = shuffle
        self.sampler = sampler
        if num_folds > 1:
            self.dataset_train = dataset_info.creator(data_path,
                                                      "train",
                                                      fold,
                                                      transforms[0],
                                                      target_transform=target_transforms[0])
            self.dataset_val = dataset_info.creator(data_path,
                                                      "valid",
                                                      fold,
                                                      transforms[1],
                                                      target_transform=target_transforms[1])
            self.dataset_test = dataset_info.creator(data_path,
                                                      "test",
                                                      fold,
                                                      transforms[2],
                                                      target_transform=target_transforms[2])
        else:
            if self.dataset_name == "audioset":
                dataset_ub = dataset_info.creator(data_path,
                                                        "train",
                                                        transforms[0],
                                                        target_transform=target_transforms[0])
                dataset_b = dataset_info.creator(os.path.join(data_path,"../audioset_b"),
                                                        "train",
                                                        transforms[0],
                                                        target_transform=target_transforms[0])
                self.dataset_train = ConcatDataset([dataset_ub,dataset_b])
            else:
                self.dataset_train = dataset_info.creator(data_path,
                                                        "train",
                                                        transforms[0],
                                                        target_transform=target_transforms[0])
            self.dataset_val = dataset_info.creator(data_path,
                                                      "valid",
                                                      transforms[1],
                                                      target_transform=target_transforms[1])
            self.dataset_test = dataset_info.creator(data_path,
                                                      "test",
                                                      transforms[2],
                                                      target_transform=target_transforms[2])
        self.save_hyperparameters(ignore="target_transforms")
    def prepare_data(self):
        pass

    def train_dataloader(self):
        # Add dataloader for ConcatDataset
        if self.dataset_name == "dcase" :
            sampler = self.dataset_train[1]["sampler"]
            batch_sizes = self.dataset_train[1]["batch_size"]
            return data.DataLoader(
                self.dataset_train[0],
                sampler=sampler,
                batch_size=batch_sizes,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
                pin_memory=True,
                )
        elif self.dataset_name == "audioset" and (self.sampler is not None):
            def worker_init_fn(id):
                # seed every worker with different seed
                # so that they don't all get the same samples for MixUp 
                rank = dist.get_rank()
                np.random.seed((id + rank +  np.random.get_state()[1][0])%(2**32))
            return data.DataLoader(
                self.dataset_train,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
                sampler= DistributedSamplerWrapper( self.sampler,range(len(self.sampler))),
                drop_last=False,
                collate_fn=self.collate_fn,
                worker_init_fn=worker_init_fn,
                pin_memory=True
                )


        else:
            return data.DataLoader(
                self.dataset_train,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=self.shuffle if self.sampler is None else False,
                sampler=self.sampler,
                drop_last=False,
                collate_fn=self.collate_fn,
                pin_memory=True
                )
                            
    def val_dataloader(self):
        return data.DataLoader(self.dataset_val,
                        batch_size=self.batch_size,
                        num_workers=self.num_workers,
                        shuffle=False,
                        sampler=None,
                        drop_last=False,
                        collate_fn=self.collate_fn,
                        pin_memory=True)
    def test_dataloader(self):
        return data.DataLoader(self.dataset_test,
                        batch_size=self.batch_size,
                        num_workers=self.num_workers,
                        shuffle=False,
                        sampler=None,
                        drop_last=False,
                        collate_fn=self.collate_fn,
                        pin_memory=True)


    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("DownstreamData")
        parser.add_argument("--dataset_name", type=str, default=None, help="dataset name")
        parser.add_argument("--data_path", type=str, default=None, help="data path")
        parser.add_argument('--batch_size_per_gpu', default=256, type=int,
            help='Per-GPU batch-size : number of distinct samples loaded on one GPU.')
        parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
        return parent_parser
