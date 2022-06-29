from pytorch_lightning import LightningModule
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from audiossl.utils.common import cosine_scheduler_step,get_params_groups
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch import nn
from audiossl.methods.frame_atst.audio_transformer import FrameAST_small,FrameAST_base
from audiossl.methods.frame_atst.byol import MultiCropWrapper,ByolLoss
import torch
import argparse

def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")

class FrameATST(nn.Module):
    def __init__(self,
                 arch="small",
                 use_cls=2,
                 symmetric=True,
                 unmask_for_cls=True,
                 crop_ratio=0.6,
                 avg_blocks=8,
                 use_mse=0,
                 **kwargs):
        super().__init__()
        if arch == "small":
            encoder_fn = FrameAST_small
            embed_dim = 384
        elif arch == "base":
            encoder_fn = FrameAST_base
            embed_dim = 768
        else:
            raise RuntimeError("arch {} is not implemented".format(arch))
        self.doublehead=use_cls>0
        self.symmetric = symmetric
        self.use_mse = use_mse
        self.student=MultiCropWrapper(encoder_fn(use_cls=use_cls,
                                                 use_unmask_for_cls=unmask_for_cls,
                                                 crop_ratio=crop_ratio,
                                                 avg_blocks=0,
                                                 use_mse=use_mse,
                                                 **kwargs),
                                      embed_dim,
                                      predictor=True,
                                      doublehead=use_cls>0,
                                      use_mse=use_mse)
        self.teacher=MultiCropWrapper(encoder_fn(use_cls=use_cls,
                                                 use_unmask_for_cls=unmask_for_cls,
                                                 crop_ratio=crop_ratio,
                                                 avg_blocks=avg_blocks,
                                                 use_mse=use_mse,
                                                 **kwargs),
                                      embed_dim,
                                      predictor=False,
                                      doublehead=use_cls>0,
                                      use_mse=use_mse)
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.load_state_dict({k:v for k,v in self.student.state_dict().items() if "predictor" not in k })
        self.loss_fn = ByolLoss(doublehead=use_cls>0,symmetric=symmetric)
    
    def forward(self,x,length,mask):
        if self.symmetric:
            if self.use_mse>0:
                tea,_ = self.teacher(x,length,mask,False)
                stu,mse_loss= self.student(x,length,mask,True)
                return list(self.loss_fn(stu,tea))+[mse_loss]
            else:
                tea = self.teacher(x,length,mask,False)
                stu = self.student(x,length,mask,True)
                return self.loss_fn(stu,tea)
            #total_loss_frm,total_loss_cls,std_frm_stu,std_frm_tea,std_cls_stu,std_cls_tea =

        else:
            if self.use_mse>0:
                tea,_ = self.teacher(x[:1],length[:1],mask[:1],False)
                stu,mse_loss = self.student(x[1:],length[1:],mask[1:],True)
                return list(self.loss_fn(stu,tea))+[mse_loss]
            else:
                tea = self.teacher(x[:1],length[:1],mask[:1],False)
                stu = self.student(x[1:],length[1:],mask[1:],True)
                return self.loss_fn(stu,tea)
            #total_loss_frm,std_frm_stu,std_frm_tea
    def update_teacher(self,m):
        with torch.no_grad():
            for param_q, param_k in zip(self.student.encoder.parameters(), self.teacher.encoder.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
            for param_q, param_k in zip(self.student.projector.parameters(), self.teacher.projector.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
            if self.doublehead:
                for param_q, param_k in zip(self.student.projector2.parameters(), self.teacher.projector2.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
        



class FrameATSTLightningModule(LightningModule):
    def __init__(self,
                 arch="small",
                 learning_rate:float=5e-4,
                 warmup_steps=1300,
                 max_steps=39000,
                 ema=0.99,
                 symmetric=True,
                 use_cls=2,
                 unmask_for_cls=True,
                 crop_ratio=0.6,
                 avg_blocks=8,
                 use_mse=0,
                 **kwargs,
                 ):
        super().__init__()
        self.model = FrameATST(arch=arch,
                               symmetric=symmetric,
                               use_cls=use_cls,
                               unmask_for_cls=unmask_for_cls,
                               crop_ratio=crop_ratio,
                               avg_blocks=avg_blocks,
                               use_mse=use_mse,
                               **kwargs)
        self.learning_rate = learning_rate 
        self.warmup_steps =  warmup_steps
        self.max_steps = max_steps
        self.symmetric=symmetric
        self.use_cls=use_cls
        self.doublehead=use_cls>0
        self.use_mse = use_mse
        self.ema_scheduler= cosine_scheduler_step(ema,1,max_steps,0)
        self.wd_scheduler = cosine_scheduler_step(0.04,0.4,max_steps,0)
        self.mylr_scheduler = cosine_scheduler_step(learning_rate,1e-6,max_steps,warmup_steps)
        self.save_hyperparameters()
    def training_step(self,batch,batch_idx):
        self.schedule()
        (melspecs,lengths,masks),_ = batch
        if self.doublehead:
            
            total_loss_frm,total_loss_cls,std_frm_stu,std_frm_tea,std_cls_stu,std_cls_tea = self.model(melspecs,lengths,masks)
            loss = total_loss_frm + total_loss_cls
            self.log("loss",loss,prog_bar=True,logger=True)
            self.log("loss_frm",total_loss_frm,prog_bar=True,logger=True)
            self.log("loss_cls",total_loss_cls,prog_bar=True,logger=True)
            self.log("std_cls_tea",std_cls_tea,prog_bar=True,logger=True)
            self.log("std_cls_stu",std_cls_stu,prog_bar=True,logger=True)
            self.log("std_frm_tea",std_frm_tea,prog_bar=True,logger=True)
            self.log("std_frm_stu",std_frm_stu,prog_bar=True,logger=True)
            self.log("ema",self.ema_scheduler[self.global_step],prog_bar=True,logger=True)
            self.log("step",self.global_step,prog_bar=True,logger=True)
        else:
            if self.use_mse>0:
                total_loss_frm,std_frm_stu,std_frm_tea,mse_loss = self.model(melspecs,lengths,masks)
                loss = total_loss_frm + mse_loss
                self.log("loss_mse",mse_loss,prog_bar=True,logger=True)
            else:
                total_loss_frm,std_frm_stu,std_frm_tea= self.model(melspecs,lengths,masks)
                loss = total_loss_frm
            self.log("loss",loss,prog_bar=True,logger=True)
            self.log("loss_frm",total_loss_frm,prog_bar=True,logger=True)
            self.log("std_frm_tea",std_frm_tea,prog_bar=True,logger=True)
            self.log("std_frm_stu",std_frm_stu,prog_bar=True,logger=True)
            self.log("ema",self.ema_scheduler[self.global_step],prog_bar=True,logger=True)
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
    def on_train_batch_end(self, outputs, batch, batch_idx: int, unused: int = 0) -> None:
        m = self.ema_scheduler[self.global_step]
        self.model.update_teacher(m)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("FrameATSTModel")
        parser.add_argument("--arch",type=str,default="small")
        parser.add_argument("--use_cls",type=int,default=2,help="""prepending cls token at last n blocks,
                                                                        default n=2; n=0 means no cls loss""")
        parser.add_argument("--unmask_for_cls",type=bool_flag,default=True,help="whether to use unmasked frames for cls loss")
        parser.add_argument("--crop_ratio",type=float,default=0.6,help="crop ratio of frames for cls loss")
        parser.add_argument("--avg_blocks",type=int,default=8,help="average n blocks of teacher network")
        parser.add_argument("--use_mse",type=int,default=0,help="values larger than 0 means using mse loss")
        parser.add_argument("--symmetric",type=bool_flag,default=True,help="whether to use symemtric loss")
        parser.add_argument("--learning_rate", default=0.0005, type=float, help="""Learning rate at the end of
            linear warmup (highest LR used during training). The learning rate is linearly scaled
            with the batch size, and specified here for a reference batch size of 256.""")
        parser.add_argument('--ema', default=0.99, type=float, help="""Base EMA
            parameter for teacher update. The value is increased to 1 during training with cosine schedule.
            """)
        parser.add_argument('--warmup_steps',default=1300,type=int)
        parser.add_argument('--max_steps',default=39010,type=int)
        return parent_parser