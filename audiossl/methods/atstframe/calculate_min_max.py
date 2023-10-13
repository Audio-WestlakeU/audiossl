import torchaudio
from torchvision import transforms
from audiossl.datasets import LMDBDataset
import sys

melspec_t = torchaudio.transforms.MelSpectrogram(
    16000, f_min=60, f_max=7800, hop_length=160, win_length=1024, n_fft=1024, n_mels=128)
to_db = torchaudio.transforms.AmplitudeToDB(stype="power",top_db=80)
transform=transforms.Compose([melspec_t,to_db])

data_path = sys.argv[1]

dset=LMDBDataset(data_path,
                            split="train",
                            subset=2000000,
                            transform=transform)


i=iter(dset)
min = 1e10
max = -1e10
n=next(i)
while n is not None:
    if n[0].min() < min:
        min=n[0].min()
    if n[0].max() > max:
        max= n[0].max()
    print(min,max)
    n=next(i)