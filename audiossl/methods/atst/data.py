from pytorch_lightning import LightningDataModule
from torch.utils import data
from audiossl.datasets import LMDBDataset
from audiossl.methods.atst.transform import ATSTTrainTransform

class ATSTDataModule(LightningDataModule):
    def __init__(self,
                 data_path=None,
                 batch_size_per_gpu=256,
                 num_workers=10,
                 subset=200000,
                 train_len=6.0,
                 **kwargs,
                 ):
        super().__init__()
        self.dataset=LMDBDataset(data_path,
                                 split="train",
                                 subset=subset,
                                 transform=ATSTTrainTransform(anchor_len=(train_len,train_len)))
        self.batch_size=batch_size_per_gpu
        self.num_workers=num_workers
        self.save_hyperparameters()
    

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
        parser.add_argument('--train_len', default=6.0, type=float, help='length of training segment')
        return parent_parser
