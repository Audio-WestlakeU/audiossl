from pytorch_lightning import LightningModule
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from audiossl.utils.common import cosine_scheduler_step,get_params_groups
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch import nn

class MAE(nn.Module):
    def __init__(self):
        super().__init__()
    def forward():
        pass

class MAELightningModule(LightningModule):
    def __init__(self,
                 arch="small",
                 learning_rate:float=5e-4,
                 warmup_steps=1300,
                 max_steps=39000,
                 **kwargs,
                 ):
        super().__init__()
        self.model = MAE(arch=arch)
        self.learning_rate = learning_rate 
        self.warmup_steps =  warmup_steps
        self.max_steps = max_steps
        self.wd_scheduler = cosine_scheduler_step(0.05,0.5,max_steps,0)
        self.mylr_scheduler = cosine_scheduler_step(learning_rate,1e-6,max_steps,warmup_steps)
        self.save_hyperparameters()
    def training_step(self,batch,batch_idx):
        self.schedule()
        (melspecs,lengths),_ = batch
        loss,std_cls_s,std_cls_t = self.model(melspecs,lengths)
        self.log("loss",loss,prog_bar=True,logger=True)
        self.log("std_cls_t",std_cls_t,prog_bar=True,logger=True)
        self.log("std_cls_s",std_cls_s,prog_bar=True,logger=True)
        self.log("step",self.global_step,prog_bar=True,logger=True)
        
        return loss
    def schedule(self):
        for i, param_group in enumerate(self.trainer.optimizers[0].param_groups):
            param_group["lr"] = self.mylr_scheduler[self.global_step]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = self.wd_scheduler[self.global_step]
        
        self.log("wd",self.wd_scheduler[self.global_step],prog_bar=True,logger=True)
        self.log("lr",param_group["lr"],prog_bar=True,logger=True)

    def configure_optimizers(self):
        optimizer = AdamW(get_params_groups(self.model.student),
                          lr=self.learning_rate,
                          weight_decay=0.)
        return [optimizer]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("MAEModel")
        parser.add_argument("--arch",type=str,default="small")
        parser.add_argument("--learning_rate", default=0.0005, type=float, help="""Learning rate at the end of
            linear warmup (highest LR used during training). The learning rate is linearly scaled
            with the batch size, and specified here for a reference batch size of 256.""")
        parser.add_argument('--warmup_steps',default=1300,type=int)
        parser.add_argument('--max_steps',default=39010,type=int)
        return parent_parser