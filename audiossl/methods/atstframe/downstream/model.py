from email.mime import audio
from pytorch_lightning import LightningModule
from audiossl.modules.head import LinearHead
from audiossl.methods.atstframe import audio_transformer
import torch
from torch import nn
from torch.nn import functional as F
from audiossl.methods.atst.downstream.utils import Metric
from audiossl.utils.common import cosine_scheduler_epoch,get_params_groups
from itertools import chain


def binary_cross_entropy_with_logits(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Calls BCE with logits and cast the target one_hot (y) encoding to floating point precision."""
    return F.binary_cross_entropy_with_logits(x, y.float())


class PretrainedEncoderPLModule(LightningModule):
    def __init__(self,
                 pretrained_encoder: audio_transformer.FrameAST,
                 chunk_len: float,
                 n_blocks: int,
                 avgpool:bool = True):
        super().__init__()
        self.encoder = pretrained_encoder
        self.chunk_len = int((self.encoder.hyper_param["anchor_len"] * 16000)/160 + 1)
        self.n_blocks = n_blocks
        self.avgpool = avgpool
        if self.encoder.nprompt>0:
            self.embed_dim = self.encoder.embed_dim*n_blocks*2
        else:
            self.embed_dim = self.encoder.embed_dim*n_blocks 

    def forward(self, batch):
        (mel, length), y = batch
        chunk_len=self.chunk_len
        total_len = mel.shape[-1]
        num_chunks = total_len // chunk_len + 1
        output=[]
        chunk_mark=[]
        for i in range(num_chunks):

            cur_len = torch.clip(length - i*chunk_len,0,chunk_len)
            if i==0:
                chunk_mark_ = cur_len > 0
            else:
                chunk_mark_ = cur_len > chunk_len//2
            start = i*chunk_len
            end = (i+1) * chunk_len
            if end > total_len:
                end = total_len
            if (end>start+20): #and (length +chunk_len//2  > end):
                mel_chunk=mel[:,:,:,start:end]
                output_chunk = self.encoder.get_intermediate_layers(mel_chunk,cur_len,n=self.n_blocks,scene=True)

                output.append(output_chunk)
                chunk_mark.append(chunk_mark_)
        chunk_mark=torch.stack(chunk_mark,dim=0).unsqueeze(-1)
        output=torch.stack(output,dim=0)
        output=torch.sum(chunk_mark*output,dim=0)/torch.sum(chunk_mark,dim=0)
        return output, y


class LinearClassifierPLModule(LightningModule):
    def __init__(self,
                 learning_rate,
                 max_epochs,
                 embed_dim,
                 num_labels,
                 multi_label=False,
                 **kwargs):
        super().__init__()
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

        self.head = LinearHead(embed_dim, num_labels)
        self.multi_label = multi_label

        if multi_label:
            self.loss_fn = binary_cross_entropy_with_logits
            self.metric = Metric(mode="mAP")
        else:
            self.loss_fn = nn.CrossEntropyLoss()
            self.metric = Metric(mode="ACC")
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        x,y = batch
        if self.multi_label == False and y.dim() > 1:
            y = y.argmax(-1)
        x = self.head(x)
        loss = self.loss_fn(x,y)
        self.log("lr",self.trainer.optimizers[0].param_groups[0]["lr"],prog_bar=True,logger=True)
        return loss

    def _cal_metric(self,output,target):
        if self.multi_label:
            self.metric.update(output.sigmoid(),target)
        else:
            self.metric.update(output,target)


    def validation_step(self, batch, batch_idx):
        x,y = batch
        if self.multi_label == False and y.dim() > 1:
            y = y.argmax(-1)
        x = self.head(x)
        loss = self.loss_fn(x,y)
        self._cal_metric(x,y)
        return loss
    def on_validation_epoch_end(self) -> None:
        metric  = self.metric.compute()
        self.metric.clear()
        self.log("val_"+self.metric.mode,metric,prog_bar=True,logger=True)


    def test_step(self, batch, batch_idx):
        x,y = batch
        if self.multi_label == False and y.dim() > 1:
            y = y.argmax(-1)
        x = self.head(x)
        loss = self.loss_fn(x,y)
        self._cal_metric(x,y)
        return loss
    def on_test_epoch_end(self) -> None:
        metric  = self.metric.compute()
        self.metric.clear()
        self.log("test_"+self.metric.mode,metric,prog_bar=True,logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.head.parameters(),
                                    self.learning_rate,
                                    momentum=0.9,
                                    weight_decay=0,
                                    )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.max_epochs, eta_min=0)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"} ]

    @staticmethod
    def add_model_specific_args(parent_parser):

        parser = parent_parser.add_argument_group("LinearClassifierModel")
        parser.add_argument("--learning_rate", default=0.002,
                            type=float, help="""Learning rate""")
        parser.add_argument('--max_epochs', default=100, type=int)
        return parent_parser

def layer_wise_lr_groups(model):

    layer_decay = model.layer_wise_lr
    num_layers = 12
    lr_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))
    model.unfreeze()

    groups = []
    for name,param in model.named_parameters():
        # encoder.mask_embed encoder.pos_embed encoder.patch_embed
        # encoder.blocks.
        # encoder.norm_frame
        # head
        name = name.replace("encoder.encoder","encoder") 
        if not param.requires_grad:
            continue

        if name.startswith("encoder.mask_embed") or name.startswith("encoder.pos_embed") or name.startswith("encoder.patch_embed"):
            if model.freeze_embed:
                groups.append({'params': param,
                            'lr_scale' : 0,})
                            #'name':name})
            else:
                groups.append({'params': param,
                            'lr_scale' : lr_scales[0],})
        elif name.startswith("encoder.blocks"):
            index = int(name.split(".")[2])
            groups.append({'params': param,
                        'lr_scale' : lr_scales[index],})
                        #'name':name})
        elif name.startswith("encoder.norm_frame"):
            groups.append({'params': param,
                        'lr_scale' : lr_scales[-2],})
                        #'name':name})
        elif name.startswith("head"):
            groups.append({'params': param,
                        'lr_scale' : lr_scales[-1],})
                        #'name':name})
        else:
            print("missed",name)
    return groups




class FineTuningPLModule(LightningModule):
    def __init__(self,
                 encoder,
                 learning_rate,
                 max_epochs,
                 niter_per_epoch,
                 warmup_epochs,
                 num_labels,
                 multi_label=False,
                 layer_wise_lr=0.75,
                 freeze_embed=False,
                 mixup_training=False,
                 optimizer="SGD",
                 **kwargs):
        super().__init__()
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.warumup_epochs = warmup_epochs
        self.niter_per_epoch = niter_per_epoch
        self.optimizer_type =optimizer

        self.encoder = encoder
        self.head = LinearHead(encoder.embed_dim, num_labels,use_norm=True, affine=False)
        self.multi_label = multi_label
        self.mixup_training = mixup_training
        self.num_labels = num_labels
        self.layer_wise_lr = layer_wise_lr
        self.freeze_embed = freeze_embed

        if multi_label or self.mixup_training:
            self.loss_fn = binary_cross_entropy_with_logits
            if multi_label:
                self.metric = Metric(mode="mAP")
            else:
                self.metric = Metric(mode="ACC")
        else:
            self.loss_fn = nn.CrossEntropyLoss()
            self.metric = Metric(mode="ACC")
        self.mylr_scheduler = cosine_scheduler_epoch(learning_rate,
                                                     1e-6,
                                                     max_epochs,
                                                     niter_per_epoch,
                                                     warmup_epochs)
        self.save_hyperparameters(ignore=["encoder"])

    def training_step(self, batch, batch_idx):
        self.encoder.train()
        self.schedule()
        x,y=self.encoder(batch)
        if self.multi_label == False and self.mixup_training == False and y.dim() > 1:
            y = y.argmax(-1)
        
        elif self.mixup_training == True and (y.dim() == 0 or y.dim() == 1) :
            y = torch.nn.functional.one_hot(y.to(torch.int64),num_classes=self.num_labels)
        
        x = self.head(x)
        loss = self.loss_fn(x,y)
        self.log("train_loss",loss,prog_bar=True,logger=True)
        return loss

    def schedule(self):
        if self.layer_wise_lr>0:
            for i, param_group in enumerate(self.trainer.optimizers[0].param_groups):
                param_group["lr"] = self.mylr_scheduler[self.global_step] * param_group["lr_scale"]
        else:
            for i, param_group in enumerate(self.trainer.optimizers[0].param_groups):
                param_group["lr"] = self.mylr_scheduler[self.global_step]
                if self.optimizer_type== "adamw":
                    if i == 0:  # only the first group is regularized
                        param_group["weight_decay"] = 5e-4
        self.log("lr",self.mylr_scheduler[self.global_step],prog_bar=True,logger=True)

    def _cal_metric(self,output,target):
        if self.multi_label:
            self.metric.update(output.sigmoid(),target)
        else:
            self.metric.update(output,target)


    def validation_step(self, batch, batch_idx):
        self.encoder.eval()
        x,y=self.encoder(batch)
        y_=y
        if self.multi_label == False and self.mixup_training == False and y.dim() > 1:
            y_ = y.argmax(-1)
        if self.mixup_training == True and (y.dim() == 0 or y.dim() == 1) :
            y_ = torch.nn.functional.one_hot(y.to(torch.int64),num_classes=self.num_labels)
        x = self.head(x)
        loss = self.loss_fn(x,y_)
        self.log("val_loss",loss,prog_bar=True,logger=True)
        self._cal_metric(x,y_.argmax(-1) if self.mixup_training and (not self.multi_label) else y_)
        return loss
    def on_validation_epoch_end(self) -> None:
        metric  = self.metric.compute()
        self.metric.clear()
        self.log("val_"+self.metric.mode,metric,prog_bar=True,logger=True)


    def test_step(self, batch, batch_idx):
        self.encoder.eval()
        x,y=self.encoder(batch)
        y_=y
        if self.multi_label == False and self.mixup_training == False and y.dim() > 1:
            y_ = y.argmax(-1)
        if self.mixup_training == True and (y.dim() == 0 or y.dim()==1) :
            y_ = torch.nn.functional.one_hot(y.to(torch.int64),num_classes=self.num_labels)
        x = self.head(x)
        loss = self.loss_fn(x,y_)
        self._cal_metric(x,y_.argmax(-1) if self.mixup_training and (not self.multi_label) else y_)
        return loss
    def on_test_epoch_end(self) -> None:
        metric  = self.metric.compute()
        self.metric.clear()
        self.log("test_"+self.metric.mode,metric,prog_bar=True,logger=True)

    def configure_optimizers(self):
        if self.optimizer_type == "SGD":
            if self.layer_wise_lr > 0 :
                optimizer = torch.optim.SGD(layer_wise_lr_groups( self),
                                            self.learning_rate,
                                            momentum=0.9,
                                            weight_decay=0,
                                            )
            else:
                optimizer = torch.optim.SGD(list(self.encoder.encoder.parameters())+list(self.head.parameters()),
                                    self.learning_rate,
                                    momentum=0.9,
                                    weight_decay=0,
                                    )
        else:
            optimizer = torch.optim.AdamW(get_params_groups( self),
                                          self.learning_rate,
                                          weight_decay=0)
        return [optimizer]
    
    @staticmethod
    def add_model_specific_args(parent_parser):

        parser = parent_parser.add_argument_group("FineTuningModel")
        parser.add_argument("--learning_rate", default=0.002,
                            type=float, help="""Learning rate""")
        parser.add_argument('--max_epochs', default=100, type=int)
        parser.add_argument('--warmup_epochs', default=5, type=int)
        return parent_parser
