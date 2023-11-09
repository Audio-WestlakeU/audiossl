import torch
import pytorch_lightning as pl
from audiossl.models.atst.audio_transformer import AST_base
from audiossl.methods.atst.downstream.utils import load_pretrained_weights
from audiossl.methods.atst.downstream.transform import FreezingTransform

class ATSTPredModule(pl.LightningModule):
    """This module has been modified for frame-level prediction"""

    def __init__(self, pretrained_ckpt_path=None, dataset_name="as_strong", **kwargs):
        super().__init__()
        self.encoder = AST_base(use_cls=True, **kwargs)
        if pretrained_ckpt_path is not None:
            load_pretrained_weights(self.encoder, pretrained_ckpt_path, checkpoint_key="teacher")
        self.embed_dim = self.encoder.embed_dim
        self.transform = FreezingTransform(max_len=10)
        self.last_layer = dataset_name != "as_strong"

    def forward(self, batch):
        (x, length), y = batch
        x = x.unsqueeze(1)
        x = self.encoder.get_intermediate_layers(
            x,
            length,
            1
        )
        x = [item[:, 1:, :] for item in x]
        x = torch.concat(x, dim=-1)
        return x, y

    def finetune_mode(self):
        if self.last_layer:
            self.freeze()
            # Unfreeze last tfm block
            for i, layer in enumerate(self.encoder.blocks):
                if i == len(self.encoder.blocks) - 1:
                    for n, p in layer.named_parameters():
                        p.requires_grad = True
            # Unfreeze last norm layer
            for n, p in self.encoder.norm.named_parameters():
                p.requires_grad = True
        else:
            for n, p in self.encoder.named_parameters():
                if "mask_embed" in n:
                    p.requires_grad = False
                else:
                    p.requires_grad = True

    def finetune_mannual_train(self):
        if self.last_layer:
            for i, layer in enumerate(self.encoder.blocks):
                if i == len(self.encoder.blocks) - 1:
                    layer.train()
            self.encoder.norm.train()
        else:
            self.train()