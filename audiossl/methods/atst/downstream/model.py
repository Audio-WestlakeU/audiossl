from email.mime import audio
from pytorch_lightning import LightningModule
from audiossl.modules.head import LinearHead
from audiossl.models.atst import audio_transformer
import torch
from torch import nn
from torch.nn import functional as F
from audiossl.methods.atst.downstream.utils import Metric


def binary_cross_entropy_with_logits(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Calls BCE with logits and cast the target one_hot (y) encoding to floating point precision."""
    return F.binary_cross_entropy_with_logits(x, y.float())


class PretrainedEncoderPLModule(LightningModule):
    def __init__(self,
                 pretrained_encoder: audio_transformer.AST,
                 chunk_len: float,
                 n_blocks: int):
        super().__init__()
        self.encoder = pretrained_encoder
        self.encoder.eval()
        self.chunk_len = int((chunk_len * 16000)/160 + 1)
        self.n_blocks = n_blocks

    def forward(self, batch):
        (x, length), y = batch
        with torch.no_grad():
            x = self.encoder.get_intermediate_layers_chunks(x,
                                                        length,
                                                        self.n_blocks,
                                                        self.chunk_len)
        return x, y


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
        self.optimizers()
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

        parser = parent_parser.add_argument_group("FreezedTransfer")
        parser.add_argument("--learning_rate", default=0.002,
                            type=float, help="""Learning rate""")
        parser.add_argument('--max_epochs', default=100, type=int)
        return parent_parser
