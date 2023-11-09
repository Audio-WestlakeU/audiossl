from pytorch_lightning import LightningDataModule
import torch
from torch.utils import data
from audiossl import datasets

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

# Add a batch sampler for DCASE concat dataset [MARK]
class DownstreamDataModule(LightningDataModule):
    def __init__(self,
                 dcase_conf:str,
                 dataset_name:str,
                 fold:int = 0,
                 batch_size_per_gpu=1024,
                 num_workers=10,
                 transforms=[None,None,None],
                 target_transforms=[None,None,None],
                 collate_fn=None,
                 limit_batch_size=None,
                 shuffle = True,
                 ignores=[],
                 **kwargs
                 ):
        super().__init__()
        self.batch_size=batch_size_per_gpu
        self.num_workers=num_workers
        self.transforms=transforms
        self.target_transforms=target_transforms
        self.collate_fn = collate_fn
        dataset_info=datasets.get_dataset(dataset_name)
        num_folds = dataset_info.num_folds
        self.num_labels=dataset_info.num_labels
        self.multi_label=dataset_info.multi_label
        self.shuffle = shuffle
        if num_folds > 1:
            self.dataset_train = dataset_info.creator(dcase_conf,
                                                      "train",
                                                      fold,
                                                      transforms[0],
                                                      target_transform=target_transforms[0])
            self.dataset_val = dataset_info.creator(dcase_conf,
                                                      "valid",
                                                      fold,
                                                      transforms[1],
                                                      target_transform=target_transforms[1])
            self.dataset_test = dataset_info.creator(dcase_conf,
                                                      "test",
                                                      fold,
                                                      transforms[2],
                                                      target_transform=target_transforms[2])
        else:
            self.dataset_train = dataset_info.creator(dcase_conf,
                                                      "train",
                                                      transforms[0],
                                                      target_transform=target_transforms[0])
            self.dataset_val = dataset_info.creator(dcase_conf,
                                                      "valid",
                                                      transforms[1],
                                                      target_transform=target_transforms[1])
            self.dataset_test = dataset_info.creator(dcase_conf,
                                                      "test",
                                                      transforms[2],
                                                      target_transform=target_transforms[2])
        self.save_hyperparameters(ignore=ignores)
    def prepare_data(self):
        pass

    def train_dataloader(self):
        # Add dataloader for ConcatDataset
        if len(self.dataset_train) > 1:
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
        else:
            return data.DataLoader(
                self.dataset_train[0],
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=True,
                drop_last=True,
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
        return data.DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            sampler=None,
            drop_last=False,
            collate_fn=self.collate_fn,
            pin_memory=True,)


    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("DownstreamData")
        parser.add_argument("--dataset_name", type=str, default=None, help="dataset name")
        parser.add_argument('--batch_size_per_gpu', default=256, type=int,
            help='Per-GPU batch-size : number of distinct samples loaded on one GPU.')
        parser.add_argument('--num_workers', default=4, type=int, help='Number of data loading workers per GPU.')
        return parent_parser
