import torch
import torchaudio.compliance.kaldi as ta_kaldi
import pytorch_lightning as pl
import torch.nn as nn
from audiossl.methods.atstframe.downstream.comparison_models.models.beats.BEATs import BEATs, BEATsConfig

audio_configs = {
    "feature_type": "fbank", 
    "sample_rate": 16000,
    "max_sample_size": 250000,
    "min_sample_size": 32000,
    "feature_rate": 100,
    "feature_dim": 128,
    "normalize": False, # must be consistent with extractor
    "deltas": False
}


class BeatsPredModule(pl.LightningModule):
    def __init__(self, pertrained_ckpt_path, dataset_name="as_strong") -> None:
        super(BeatsPredModule, self).__init__()
        checkpoint = torch.load(pertrained_ckpt_path)
        cfg = BEATsConfig(checkpoint["cfg"])
        # cfg.set("layer_wise_gradient_decay_ratio", 0.75)
        cfg.set("encoder_layerdrop", 0.0)
        self.encoder = BEATs(cfg)
        self.encoder.load_state_dict(checkpoint["model"])
        self.feat_mean = nn.AvgPool1d(8, 8)
        self.embed_dim = 768
        self.opt_len = [len(self.encoder.encoder.layers) - (i + 1) for i in range(3)]
        self.last_layer = dataset_name != "as_strong"


    def forward(self, batch):
        (x, length), y = batch
        x = self.encoder.extract_features(x, None)[0]
        x = self.feat_mean(x.transpose(-1, -2)).transpose(-1, -2)
        return x, y

    def transform(self,
                  wav, 
                  fbank_mean: float = 15.41663,
                  fbank_std: float = 6.55582,):
        waveform = wav.unsqueeze(0) * 2 ** 15
        fbank = ta_kaldi.fbank(waveform, num_mel_bins=128, sample_frequency=16000, frame_length=25, frame_shift=10)
        fbank = (fbank - fbank_mean) / (2 * fbank_std)
        return fbank, fbank.shape[0]
    
    def finetune_mode(self):
        if self.last_layer:
            self.freeze()
            # Unfreeze last tfm block
            for i, layer in enumerate(self.encoder.encoder.layers):
                if i == len(self.encoder.encoder.layers) - 1:
                    print("unfreeze:", i)
                    for n, p in layer.named_parameters():
                        if "relative_attention_bias" not in n:
                            p.requires_grad = True
        else:
            for p in self.encoder.parameters():
                p.requires_grad = True

    def finetune_mannual_train(self):
        if self.last_layer:
            for i, layer in enumerate(self.encoder.encoder.layers):
                if i == len(self.encoder.encoder.layers) - 1:
                    layer.train()
        else:
            self.train()

def calculate_stat(path_1, path_2):
    from torch.utils.data import DataLoader
    from glob import glob
    from audiossl.datasets.dcase_utils.datasets import read_audio
    from sklearn.utils.extmath import _incremental_mean_and_var
    from tqdm import tqdm
    import numpy as np
    import torch

    class ReadWavDataset(torch.utils.data.Dataset):
        def __init__(self, path):
            self.file_names = glob(path + "*.wav")
        
        def __len__(self):
            return len(self.file_names)

        def __getitem__(self, idx):
            file = self.file_names[idx]
            wav, _, _, _ = read_audio(file, random_channel=False, multisrc=False, pad_to=160000)
            return wav        

    dataloader = DataLoader(
        dataset=ReadWavDataset(path=path_1),
        batch_size=128,
        num_workers=12,
    )

    # filenames = filenames[:100]
    last_mean = 0
    last_std = 0
    last_count = 0
    for wavs in tqdm(dataloader):
        wavs = wavs.cuda()
        fbanks = []
        for wav in wavs:
            waveform = wav.unsqueeze(0) * 2 ** 15
            fbank = ta_kaldi.fbank(waveform, num_mel_bins=128, sample_frequency=16000, frame_length=25, frame_shift=10)
            fbanks.append(fbank)
        fbank = torch.stack(fbanks, dim=0)
        last_mean, last_std, last_count = _incremental_mean_and_var(fbank.reshape(-1, 1).detach().cpu().numpy(), last_mean, last_std, last_count)
        print(last_mean, np.sqrt(last_std))
    print(last_mean, np.sqrt(last_std))
    return last_mean, last_std