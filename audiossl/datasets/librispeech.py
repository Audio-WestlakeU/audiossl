
import torchaudio
import sys
from torch.utils.data import ConcatDataset



class LibriSpeechDataset:
    def __init__(self,path,transform=None):
        ds1=torchaudio.datasets.LIBRISPEECH(path,url="train-clean-100",download=False)
        ds2=torchaudio.datasets.LIBRISPEECH(path,url="train-clean-360",download=False)
        ds3=torchaudio.datasets.LIBRISPEECH(path,url="train-other-500",download=False)
        self._ds = ConcatDataset([ds1,ds2,ds3])
        self.transform=transform
    def __getitem__(self,index):
        item = self._ds[index]
        wav,sr = item[0],item[1]
        if self.transform is not None:
            return self.transform(wav),0
        else:
            return wav,0
    def __len__(self):
        return len(self._ds)

