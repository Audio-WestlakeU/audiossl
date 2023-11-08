import csv
import json
import torchaudio
import numpy as np
import torch
import torch.nn.functional
import torch.nn as nn
import pytorch_lightning as pl
from audiossl.methods.atst.downstream.utils_dcase.comparison_models.ast_model import ASTModel
import math

# This file contains the utilities of SSAST freezing test, including:
# Data prepocessing (wav2fbank, normalization)
audio_configs = {
    "n_mels": 128,
    "sr": 16000,
    "norm_mean": -6.030435443767988,
    "norm_std": 4.102992546322562,
}

class PatchASTModel(ASTModel):
    def __init__(self, fstride=10, tstride=10, input_fdim=128, input_tdim=998, imagenet_pretrain=True, audioset_pretrain=True, model_size='base384', verbose=True):
        super(__class__, self).__init__(
            fstride=fstride, 
            tstride=tstride, 
            input_fdim=input_fdim, 
            input_tdim=input_tdim, 
            imagenet_pretrain=imagenet_pretrain, 
            audioset_pretrain=audioset_pretrain, 
            model_size=model_size, 
            verbose=verbose)
        self.feat_mean = nn.AvgPool2d([12, 1], padding=[1, 0])

    def forward(self, x):
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)

        B = x.shape[0]
        x = self.v.patch_embed(x)
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)
        x = x[:, 2:, :].reshape(B, 12, -1, 768)
        x = x.permute(0, 3, 1, 2)
        x = self.feat_mean(x)
        x = x.squeeze(-2).transpose(1, 2)
        x = torch.concat([x, x[:, -1, :].unsqueeze(1)], dim=1)
        return x

class PatchASTPredModule(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = PatchASTModel()
        self.embed_dim = 768
        
    def forward(self, batch):
        (x, length), y = batch
        x = self.encoder(x)
        return x, y

    @staticmethod
    def transform(wav):
        wav = (wav - wav.mean()).unsqueeze(0)   # add fake channel
        # LogFBank
        fbank = torchaudio.compliance.kaldi.fbank(
            wav, 
            htk_compat=True, 
            sample_frequency=audio_configs["sr"], 
            use_energy=False, 
            window_type='hanning', 
            num_mel_bins=audio_configs["n_mels"], 
            dither=0.0, 
            frame_shift=10
            )
        fbank = (fbank - audio_configs['norm_mean']) / (audio_configs['norm_std'] * 2)

        return fbank, fbank.shape[0]

    def finetune_mode(self):
        self.freeze()
        # Unfreeze last tfm block
        for i, layer in enumerate(self.encoder.v.blocks):
            if i == len(self.encoder.v.blocks) - 1:
                for n, p in layer.named_parameters():
                    p.requires_grad = True
        # Unfreeze last norm layer
        for n, p in self.encoder.v.norm.named_parameters():
            p.requires_grad = True

    def finetune_mannual_train(self):
        for i, layer in enumerate(self.encoder.v.blocks):
            if i == len(self.encoder.v.blocks) - 1:
                layer.train()
        self.encoder.v.norm.train()



if __name__ == "__main__":
    fake_wav = torch.rand([160000])
    module = PatchASTPredModule()
    fake_fb, _ = module.transform(fake_wav)
    fake_batch = fake_fb.unsqueeze(0).expand(10, -1, -1)
    fake_output, _ = module(((fake_batch, 0), 0))
    print(fake_output.shape)
    
    module.finetune_mode()
    module.finetune_mannual_train()