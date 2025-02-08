import torch
import pytorch_lightning as pl
from audiossl.methods.atstframe.downstream.comparison_models.models.frame_atst import FrameAST_base, FrameATSTLightningModule
from audiossl.methods.atstframe.downstream.utils import load_pretrained_weights
from audiossl.methods.atstframe.downstream.transform import FreezingTransform

class FrameATSTPredModule(pl.LightningModule):
    """This module has been modified for frame-level prediction"""

    def __init__(self, pretrained_ckpt_path, finetune_layer='all', use_last=True, **kwargs):
        super().__init__()
        self.encoder = get_frame_atst(pretrained_ckpt_path, **kwargs)
        self.embed_dim = self.encoder.embed_dim
        self.transform = FreezingTransform(max_len=10)
        self.finetune_layer = finetune_layer
        self.use_last = use_last
        self.layer_weights = torch.nn.parameter.Parameter(data=torch.ones(12), requires_grad=True)

    def forward(self, batch):
        (x, length), y = batch
        x = x.unsqueeze(1)
        use_n_layer_weights = 1 if self.use_last else 12
        x = self.encoder.get_intermediate_layers(
            x,
            length,
            use_n_layer_weights,
            scene=False
        )
        if self.use_last:
            return x.squeeze(-1), y
        else:  # weighted sum of all layers
            weights = torch.softmax(self.layer_weights, dim=0)
            weighted_x = torch.matmul(x, weights)  # matrix multiply
            return weighted_x, y

    def finetune_mode(self):
        if self.finetune_layer == 'last_layer':
            print("Finetune last layer.")
            self.freeze()
            # Unfreeze last tfm block
            for i, layer in enumerate(self.encoder.blocks):
                if i == len(self.encoder.blocks) - 1:
                    for n, p in layer.named_parameters():
                        p.requires_grad = True
            # Unfreeze last norm layer
            for n, p in self.encoder.norm_frame.named_parameters():
                p.requires_grad = True
        elif self.finetune_layer == 'all':
            print("Finetune all layers.")
            for n, p in self.encoder.named_parameters():
                if "mask_embed" in n:
                    p.requires_grad = False
                else:
                    p.requires_grad = True
        else:
            raise NotImplementedError(f"Finetune mode: {self.finetune_layer} not supported!")

    def finetune_mannual_train(self):
        if self.finetune_layer == 'last_layer':   # for DESED dataset。根据论文，只unfreeze最后一个encoder block, 原因是数据量少，防止overfitting
            for i, layer in enumerate(self.encoder.blocks):
                if i == len(self.encoder.blocks) - 1:
                    layer.train()
            self.encoder.norm_frame.train()
        elif self.finetune_layer == 'all':        # as_strong dataset，确认我的case是这个, 可以对比跑一下上面那个试试
            self.encoder.train()
        else:
            raise NotImplementedError(f"Finetune mode: {self.finetune_layer} not supported!")

def get_frame_atst(pretrained_ckpt_path, **kwargs):
    # get pretrained encoder
    s = torch.load(pretrained_ckpt_path, map_location="cpu")
    param_names = list(s['state_dict'].keys())
    if not ('model' in param_names[0]):
        print("Loading frame-atst model from initialize FrameATSTLightningModule")
        pretrained_model = FrameATSTLightningModule(**kwargs)
        renamed_state_dict = {}
        for k, v in s['state_dict'].items():
            if "encoder.encoder." in k:
                renamed_state_dict[k.replace("encoder.encoder.", "")] = v
        pretrained_model.model.teacher.encoder.load_state_dict(renamed_state_dict)
    else:
        print("Loading frame-atst model from checkpoint:", pretrained_ckpt_path)
        pretrained_model = FrameATSTLightningModule.load_from_checkpoint(pretrained_ckpt_path, **kwargs)
    pretrained_encoder = pretrained_model.model.teacher.encoder
    pretrained_encoder.hyper_param = s['hyper_parameters']
    return pretrained_encoder