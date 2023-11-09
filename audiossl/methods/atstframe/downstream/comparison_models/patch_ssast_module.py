import torchaudio
import torch
import torch.nn.functional
import torch.nn as nn
import pytorch_lightning as pl
from audiossl.methods.atstframe.downstream.comparison_models.models import ASTModel

# This file contains the utilities of SSAST freezing test, including:
# Data prepocessing (wav2fbank, normalization)
audio_configs = {
    "n_mels": 128,
    "sr": 16000,
    "norm_mean": -6.030435443767988,
    "norm_std": 4.102992546322562,
}
 
class SSASTModel(ASTModel):
    def __init__(self, label_dim=527, fshape=128, tshape=2, fstride=128, tstride=2, input_fdim=128, input_tdim=1024, model_size='base', pretrain_stage=True, load_pretrained_mdl_path=None):
        super(SSASTModel, self).__init__(label_dim, fshape, tshape, fstride, tstride, input_fdim, input_tdim, model_size, pretrain_stage, load_pretrained_mdl_path)
        self.feat_mean = nn.AvgPool2d([8, 1], padding=[1, 0])

    def forward(self, x):
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)
        # Rewrite ASTModel forward
        B = x.shape[0]
        x = self.v.patch_embed(x)
        if self.cls_token_num == 2:
            cls_tokens = self.v.cls_token.expand(B, -1, -1)
            dist_token = self.v.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_token, x), dim=1)
        else:
            cls_tokens = self.v.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk_id, blk in enumerate(self.v.blocks):
            x = blk(x)
        x = self.v.norm(x)
        x = x[:, self.cls_token_num:, :].reshape(B, 8, -1, 768)
        # average output of all tokens except cls token(s)
        x = x.permute(0, 3, 1, 2)
        x = self.feat_mean(x)
        x = x.squeeze(-2).transpose(1, 2)
        return x

class PatchSSASTPredModule(pl.LightningModule):
    def __init__(self, pretrained_ckpt_path, dataset_name="as_strong") -> None:
        super().__init__()
        self.encoder = SSASTModel(label_dim=1, fshape=16, tshape=16, fstride=16, tstride=16, 
                                 input_fdim=128, input_tdim=998, model_size="base", pretrain_stage=False,
                                 load_pretrained_mdl_path=pretrained_ckpt_path)
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

        return fbank, fbank.shape[0]

    def finetune_mode(self):
        if self.last_layer:
            self.freeze()
            # Unfreeze last tfm block
            for i, layer in enumerate(self.encoder.v.blocks):
                if i == len(self.encoder.v.blocks) - 1:
                    for n, p in layer.named_parameters():
                        p.requires_grad = True
            # Unfreeze last norm layer
            for n, p in self.encoder.v.norm.named_parameters():
                p.requires_grad = True
        else:
            for n, p in self.named_parameters():
                if (".v.head" in n) or (".mlp_head." in n):
                    p.requires_grad = False
                else:
                    p.requires_grad = True
    def finetune_mannual_train(self):
        if self.last_layer:
            for i, layer in enumerate(self.encoder.v.blocks):
                if i == len(self.encoder.v.blocks) - 1:
                    layer.train()
            self.encoder.v.norm.train()        
        else:
            self.train()



def calculate_stat(path_1, path_2):
    from glob import glob
    from audiossl.datasets.dcase_utils.datasets import read_audio
    from tqdm import tqdm

    running_stats = []
    filenames = glob(path_1 + "*.wav") + glob(path_2 + "*.wav")
    # filenames = filenames[:100]
    element = 0
    for file in tqdm(filenames):
        wav, _, _, _ = read_audio(file, random_channel=False, multisrc=False, pad_to=None)
        wav = (wav - wav.mean()).unsqueeze(0)
        melspec = torchaudio.compliance.kaldi.fbank(
            wav, 
            htk_compat=True, 
            sample_frequency=audio_configs["sr"], 
            use_energy=False, 
            window_type='hanning', 
            num_mel_bins=audio_configs["n_mels"], 
            dither=0.0, 
            frame_shift=10
            )
        running_stats.append(melspec)
        element += melspec.numel()
    # calculate mean
    running_mean = 0
    for emd in running_stats:
        running_mean += emd.sum().item() / element

    running_std = 0
    for emd in running_stats:
        running_std += ((emd - running_mean) ** 2).sum().item() / element
    running_std = running_std
    return running_mean, running_std

