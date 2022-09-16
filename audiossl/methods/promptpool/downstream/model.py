from email.mime import audio
from pytorch_lightning import LightningModule
from audiossl.modules.head import LinearHead
from audiossl.methods.promptpool import audio_transformer,prompt_tuning,prompt_pool
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
        self.chunk_len = int((chunk_len * 16000)/160 + 1)
        self.n_blocks = n_blocks
        self.avgpool = avgpool
        self.embed_dim = self.encoder.embed_dim*n_blocks

    def forward(self, batch):
        (mel, length), y = batch
        chunk_len=601
        total_len = mel.shape[-1]
        num_chunks = total_len // chunk_len + 1
        output=[]
        for i in range(num_chunks):

            start = i*chunk_len
            end = (i+1) * chunk_len
            if end > total_len:
                end = total_len
            if (end>start): #and (length +chunk_len//2  > end):
                mel_chunk=mel[:,:,:,start:end]
                len_chunk = mel_chunk.shape[-1] #if length>end+chunk_len else (length - end)
                len_chunk = torch.tensor([len_chunk]).cuda().expand(mel.shape[0])
                output_chunk = self.encoder.get_intermediate_layers(mel_chunk,len_chunk,n=self.n_blocks)

                output.append(output_chunk)
        output=torch.stack(output,dim=0)
        output=torch.mean(output,dim=0)
        return output, y

class PretrainedCLsPromptEncoderPLModule(LightningModule):
    def __init__(self,
                 pretrained_encoder: prompt_tuning.ClsAST,
                 chunk_len: float,
                 n_blocks: int,
                 avgpool:bool = True):
        super().__init__()
        self.encoder = pretrained_encoder
        self.chunk_len = int((chunk_len * 16000)/160 + 1)
        self.n_blocks = n_blocks
        self.avgpool = avgpool
        self.embed_dim = self.encoder.embed_dim*2*n_blocks

    def forward(self, batch):
        (mel, length), y = batch
        chunk_len=601
        total_len = mel.shape[-1]
        num_chunks = total_len // chunk_len + 1
        output_cls=[]
        output_frame=[]
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
            if (end>start): #and (length +chunk_len//2  > end):
                mel_chunk=mel[:,:,:,start:end]
                len_chunk = mel_chunk.shape[-1] #if length>end+chunk_len else (length - end)
                len_chunk = torch.tensor([len_chunk]).cuda().expand(mel.shape[0])
                output_cls_chunk,output_frame_chunk = self.encoder.get_last_n_blocks(mel_chunk,cur_len,n=self.n_blocks)

                output_cls.append(output_cls_chunk)
                output_frame.append(output_frame_chunk)
                chunk_mark.append(chunk_mark_)
        chunk_mark=torch.stack(chunk_mark,dim=0).unsqueeze(-1)
        output_cls=torch.stack(output_cls,dim=0)
        output_cls = torch.sum(chunk_mark*output_cls,dim=0)/torch.sum(chunk_mark,dim=0)
        output_frame=torch.stack(output_frame,dim=0)
        output_frame = torch.sum(chunk_mark*output_frame,dim=0)/torch.sum(chunk_mark,dim=0)
        output = torch.cat([output_cls,output_frame],dim=1)
        return output, y


class PretrainedPromptPoolEncoderPLModule(LightningModule):
    def __init__(self,
                 pretrained_encoder: prompt_pool.PromptPoolAST,
                 queries,
                 chunk_len: float,
                 n_blocks: int,
                 avgpool:bool = True,
                 samplewise_query:bool=False):
        super().__init__()
        self.encoder = pretrained_encoder
        self.queries = queries
        self.chunk_len = int((chunk_len * 16000)/160 + 1)
        self.n_blocks = n_blocks
        self.avgpool = avgpool
        self.embed_dim = self.encoder.embed_dim*2*n_blocks
        self.samplewise_query=samplewise_query

    def forward(self, batch):
        (mel, length),  y, key = batch
        if self.samplewise_query:
            queries = torch.mean(self.queries_train,dim=0,keepdim=True).expand(mel.shape[0],-1).to("cuda")
        else:
            queries = []
            for k in key:
                queries.append(self.queries[k])
            queries = torch.stack(queries,dim=0).to("cuda")
        chunk_len=601
        total_len = mel.shape[-1]
        num_chunks = total_len // chunk_len + 1
        output_cls=[]
        output_frame=[]
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
            if (end>start): #and (length +chunk_len//2  > end):
                mel_chunk=mel[:,:,:,start:end]
                len_chunk = mel_chunk.shape[-1] #if length>end+chunk_len else (length - end)
                len_chunk = torch.tensor([len_chunk]).cuda().expand(mel.shape[0])
                #query =self.query.to('cuda').unsqueeze(0).expand(mel_chunk.shape[0],-1)
                output_cls_chunk,output_frame_chunk = self.encoder.get_last_n_blocks(mel_chunk,cur_len,queries,n=self.n_blocks)

                output_cls.append(output_cls_chunk)
                output_frame.append(output_frame_chunk)
                chunk_mark.append(chunk_mark_)
        chunk_mark=torch.stack(chunk_mark,dim=0).unsqueeze(-1)
        output_cls=torch.stack(output_cls,dim=0)
        output_cls = torch.sum(chunk_mark*output_cls,dim=0)/torch.sum(chunk_mark,dim=0)
        output_frame=torch.stack(output_frame,dim=0)
        output_frame = torch.sum(chunk_mark*output_frame,dim=0)/torch.sum(chunk_mark,dim=0)
        output = torch.cat([output_cls,output_frame],dim=1)
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
            self.metric.update(output.sigmoid().cpu().numpy(),target.cpu().numpy())
        else:
            self.metric.update(output.cpu(),target.cpu())


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

class FineTuningPLModule(LightningModule):
    def __init__(self,
                 encoder,
                 learning_rate,
                 max_epochs,
                 niter_per_epoch,
                 warmup_epochs,
                 num_labels,
                 multi_label=False,
                 mixup_training=False,
                 **kwargs):
        super().__init__()
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.warumup_epochs = warmup_epochs
        self.niter_per_epoch = niter_per_epoch

        self.encoder = encoder
        self.head = LinearHead(encoder.embed_dim, num_labels,use_norm=True, affine=False)
        self.multi_label = multi_label
        self.mixup_training = mixup_training
        self.num_labels = num_labels

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
        
        x = self.head(x)
        loss = self.loss_fn(x,y)
        self.log("lr",self.trainer.optimizers[0].param_groups[0]["lr"],prog_bar=True,logger=True)
        self.log("train_loss",loss,prog_bar=True,logger=True)
        self.optimizers()
        return loss

    def schedule(self):
        for i, param_group in enumerate(self.trainer.optimizers[0].param_groups):
            param_group["lr"] = self.mylr_scheduler[self.global_step]
        self.log("lr",param_group["lr"],prog_bar=True,logger=True)

    def _cal_metric(self,output,target):
        if self.multi_label:
            self.metric.update(output.sigmoid().cpu().numpy(),target.cpu().numpy())
        else:
            self.metric.update(output.cpu(),target.cpu())


    def validation_step(self, batch, batch_idx):
        self.encoder.eval()
        x,y=self.encoder(batch)
        y_=y
        if self.multi_label == False and self.mixup_training == False and y.dim() > 1:
            y_ = y.argmax(-1)
        if self.mixup_training == True and y.dim() == 0 :
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
        if self.mixup_training == True and y.dim() == 0 :
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
        optimizer = torch.optim.SGD(list(self.encoder.encoder.parameters())+list(self.head.parameters()),
                                    self.learning_rate,
                                    momentum=0.9,
                                    weight_decay=0,
                                    )
        return [optimizer]

    @staticmethod
    def add_model_specific_args(parent_parser):

        parser = parent_parser.add_argument_group("FineTuningModel")
        parser.add_argument("--learning_rate", default=0.002,
                            type=float, help="""Learning rate""")
        parser.add_argument('--max_epochs', default=100, type=int)
        parser.add_argument('--warmup_epochs', default=5, type=int)
        return parent_parser
