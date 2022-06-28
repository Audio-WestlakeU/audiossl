from pytorch_lightning import LightningModule
from audiossl.methods.frame_atst.prompt_tuning import ClsAST
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from audiossl.utils.common import cosine_scheduler_step,get_params_groups
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch import nn
from audiossl.models.atst.byol import MultiCropWrapper,ByolLoss
import torch


def get_params_groups_custom(model,filter="prompts"):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        print(name)
        if not param.requires_grad:
            continue
        #if "framemodel" in name:
        #    continue
        print(name,"train")
        not_regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]

class ClsPrompt(nn.Module):
    def __init__(self,framemodel,nprompt,pool):
        super().__init__()
        encoder_fn=ClsAST
        

        self.nprompt=nprompt
        self.pool=pool
        student=encoder_fn(framemodel,nprompt,pool)
        self.embed_dim= student.framemodel.embed_dim if self.pool == "mean" else self.nprompt*student.framemodel.embed_dim 


        self.student=MultiCropWrapper(student,
                                      self.embed_dim,
                                      predictor=True
                                      )
        self.teacher=MultiCropWrapper(encoder_fn(framemodel,
                                                 nprompt,
                                                 pool),
                                      self.embed_dim,
                                      predictor=False
                                      )
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.load_state_dict({k:v for k,v in self.student.state_dict().items() if "predictor" not in k })
        self.loss_fn = ByolLoss(2)
    def forward(self,x,length):
        stu=self.student(x,length)
        tea=self.teacher(x,length)
        return self.loss_fn(stu,tea)

    def update_teacher(self,m):
        with torch.no_grad():
            for param_q, param_k in zip(self.student.encoder.parameters(), self.teacher.encoder.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
            for param_q, param_k in zip(self.student.projector.parameters(), self.teacher.projector.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)


class ClsPromptLightningModule(LightningModule):
    def __init__(self,
                 framemodel,
                 nprompt=3,
                 pool="mean",
                 learning_rate:float=5e-4,
                 warmup_steps=1300,
                 max_steps=39000,
                 ema=0.99,
                 **kwargs,
                 ):
        super().__init__()
        self.model = ClsPrompt(framemodel,nprompt,pool)
        self.learning_rate = learning_rate 
        self.warmup_steps =  warmup_steps
        self.max_steps = max_steps
        self.ema_scheduler= cosine_scheduler_step(ema,1,max_steps,0)
        self.wd_scheduler = cosine_scheduler_step(0.04,0.4,max_steps,0)
        self.mylr_scheduler = cosine_scheduler_step(learning_rate,1e-6,max_steps,warmup_steps)
        self.save_hyperparameters()
    def training_step(self,batch,batch_idx):
        self.schedule()
        (melspecs,lengths),_ = batch
        loss,std_cls_s,std_cls_t = self.model(melspecs,lengths)
        self.log("loss",loss,prog_bar=True,logger=True)
        self.log("std_cls_t",std_cls_t,prog_bar=True,logger=True)
        self.log("std_cls_s",std_cls_s,prog_bar=True,logger=True)
        self.log("ema",self.ema_scheduler[self.global_step],prog_bar=True,logger=True)
        self.log("step",self.global_step,prog_bar=True,logger=True)
        
        return loss
    def schedule(self):
        for i, param_group in enumerate(self.trainer.optimizers[0].param_groups):
            param_group["lr"] = self.mylr_scheduler[self.global_step]

            """
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = self.wd_scheduler[self.global_step]
            """
        
        self.log("wd",self.wd_scheduler[self.global_step],prog_bar=True,logger=True)
        self.log("lr",param_group["lr"],prog_bar=True,logger=True)

    def configure_optimizers(self):
        optimizer = AdamW(get_params_groups(self.model.student),
                          lr=self.learning_rate,
                          weight_decay=0.)
        return [optimizer]
    def on_train_batch_end(self, outputs, batch, batch_idx: int, unused: int = 0) -> None:
        m = self.ema_scheduler[self.global_step]
        self.model.update_teacher(m)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("ClsPromptModel")
        parser.add_argument("--framemodel",type=str,help="""path of frame model checkpoint""")
        parser.add_argument("--nprompt",type=int,default=3)
        parser.add_argument("--pool",type=str,default="mean")
        parser.add_argument("--learning_rate", default=0.0005, type=float, help="""Learning rate at the end of
            linear warmup (highest LR used during training). The learning rate is linearly scaled
            with the batch size, and specified here for a reference batch size of 256.""")
        parser.add_argument('--ema', default=0.99, type=float, help="""Base EMA
            parameter for teacher update. The value is increased to 1 during training with cosine schedule.
            """)
        parser.add_argument('--warmup_steps',default=1300,type=int)
        parser.add_argument('--max_steps',default=39010,type=int)
        return parent_parser