import csv
import json
import torchaudio
import numpy as np
import torch
import torch.nn.functional
import torch.nn as nn
import pytorch_lightning as pl
from audiossl.methods.atstframe.downstream.comparison_models.models.audioMAE_model import vit_base_patch16, PatchEmbed_new

# This file contains the utilities of SSAST freezing test, including:
# Data prepocessing (wav2fbank, normalization)
audio_configs = {
    "n_mels": 128,
    "sr": 16000,
    "norm_mean": -6.030435443767988,
    "norm_std": 4.102992546322562,
}
 
class AudioMAEModel(pl.LightningModule):
    def __init__(self, pretrained_path):
        super(AudioMAEModel, self).__init__()
        # Load pre-trained model
        self.encoder = vit_base_patch16()
        self.encoder.patch_embed = PatchEmbed_new(img_size=(1024, 128), patch_size=(16,16), in_chans=1, embed_dim=768, stride=16) # no overlap. stride=img_size=16
        num_patches = self.encoder.patch_embed.num_patches
        #num_patches = 512 # assume audioset, 1024//16=64, 128//16=8, 512=64x8
        self.encoder.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, 768), requires_grad=False)  # fixed sin-cos embedding

        
        checkpoint = torch.load(pretrained_path, map_location="cpu")
        print(f"Load pre-trained checkpoint from: {pretrained_path}" )
        checkpoint_model = checkpoint['model']
        state_dict = self.encoder.state_dict()

        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        # load pre-trained model
        self.encoder.load_state_dict(checkpoint_model, strict=False)
        self.feat_mean = nn.AvgPool1d(8, 8)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.encoder.patch_embed(x)
        B, T, _ = x.shape
        x = x + self.encoder.pos_embed[:, 1: T + 1, :]
        cls_token = self.encoder.cls_token + self.encoder.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)        
        x = self.encoder.pos_drop(x)

        # apply Transformer blocks
        for blk in self.encoder.blocks:
            x = blk(x)
        
        x = x[:, 1:, :]
        x = self.encoder.norm(x)
        x = self.feat_mean(x.transpose(-1, -2)).transpose(-1, -2)
        
        return x

class AudioMAEPredModule(pl.LightningModule):
    def __init__(self, pretrained_path, dataset_name="as_strong") -> None:
        super().__init__()
        self.encoder = AudioMAEModel(pretrained_path)
        self.embed_dim = 768
        self.last_layer = dataset_name != "as_strong"
        
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

        return fbank, fbank.shape[1]

    def finetune_mode(self):
        if self.last_layer:
            self.freeze()
            # Unfreeze last tfm block
            for i, layer in enumerate(self.encoder.encoder.blocks):
                if i == len(self.encoder.encoder.blocks) - 1:
                    for n, p in layer.named_parameters():
                        p.requires_grad = True
        else:
            # Unfreeze last norm layer
            for n, p in self.encoder.encoder.norm.named_parameters():
                p.requires_grad = True
            for n, p in self.encoder.named_parameters():
                if ".head." in n:
                    p.requires_grad = False
                else:
                    p.requires_grad = True

    def finetune_mannual_train(self):
        if self.last_layer:
            for i, layer in enumerate(self.encoder.encoder.blocks):
                if i == len(self.encoder.encoder.blocks) - 1:
                    layer.train()
            self.encoder.encoder.norm.train()
        else:
            self.train()