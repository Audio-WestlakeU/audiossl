from pytorch_lightning import LightningDataModule
from torch.utils import data
from audiossl.datasets import LMDBDataset,LibriSpeechDataset
from transform import FrameATSTTrainTransform
import argparse
def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


class FrameATSTDataModule(LightningDataModule):
    def __init__(self,
                 data_path=None,
                 batch_size_per_gpu=256,
                 num_workers=10,
                 subset=200000,
                 win_length=1024,
                 aug_tea=True,
                 aug_stu=True,
                 freq_wrap=True,
                 mix_up=True,
                 mask_ratio=0.75,
                 mask_type="block",
                 anchor_len=6.,
                 mask_len=5,
                 min_mask_len=2,
                 n_mels=64,
                 **kwargs,
                 ):
        super().__init__()
        import os
        from torch.utils.data import ConcatDataset

        dataset_ub=LMDBDataset(data_path,
                                 split="train",
                                 subset=subset,
                                 transform=FrameATSTTrainTransform(
                                                                   win_length=win_length,
                                                                   aug_tea=aug_tea,
                                                                   aug_stu=aug_stu,
                                                                   freq_wrap=freq_wrap,
                                                                   mask_ratio=mask_ratio,
                                                                   anchor_len=anchor_len,
                                                                   mask_type=mask_type,
                                                                   mask_len=mask_len,
                                                                   min_mask_len=min_mask_len,
                                                                   n_mels=n_mels,
                                                                   **kwargs))

        # we only use unbalanced set for self supervised pretraining
        """
        dataset_ls = LibriSpeechDataset(data_path,
                                        transform=FrameATSTTrainTransform(
                                                                   win_length=win_length,
                                                                   aug_tea=aug_tea,
                                                                   aug_stu=aug_stu,
                                                                   mix_up=mix_up,
                                                                   freq_wrap=freq_wrap,
                                                                   mask_ratio=mask_ratio,
                                                                   anchor_len=anchor_len,
                                                                   mask_type=mask_type,
                                                                   mask_len=mask_len,
                                                                   min_mask_len=min_mask_len,
                                                                   n_mels=n_mels,
                                                                   **kwargs))
        
        """
        self.dataset = dataset_ub
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
        parser = parent_parser.add_argument_group("FrameATSTData")
        parser.add_argument("--data_path", type=str, default=None, help="data path")
        parser.add_argument('--batch_size_per_gpu', default=256, type=int,
            help='Per-GPU batch-size : number of distinct samples loaded on one GPU.')
        parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
        parser.add_argument('--subset', default=200000, type=int, help='subset of training data')
        parser.add_argument('--win_length', default=1024, type=int, help='windown length')
        parser.add_argument('--aug_tea', default=True, type=bool_flag, help='whether to augment the view fed into teacher branch; if symmetric is True, this augmented view is fed into both teacher and student.')
        parser.add_argument('--aug_stu', default=True, type=bool_flag, help='whether to augment the view fed into teacher branch; if symmetric is True, this augmented view is fed into both teacher and student.')
        parser.add_argument('--freq_wrap', default=True, type=bool_flag, help='freq wraping or not')
        parser.add_argument('--mix_up', default=True, type=bool_flag, help='mixup or not')
        parser.add_argument('--anchor_len',default=6.,type=float,help="length of training samples")
        parser.add_argument('--mask_ratio',default=0.75,type=float,help="masking ratio")
        parser.add_argument('--mask_len',default=5,type=int,help="masking block length")
        parser.add_argument('--min_mask_len',default=2,type=int,help="minimum masking block length")
        parser.add_argument('--n_mels',default=64,type=int,help="number of mel channels")
        parser.add_argument('--mask_type',default="block",type=str,help="masking type: random or block")

        return parent_parser
