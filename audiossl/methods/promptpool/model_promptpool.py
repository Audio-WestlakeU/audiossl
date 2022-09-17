from tkinter import W
from pytorch_lightning import LightningModule
from audiossl.methods.promptpool.prompt_pool import PromptPoolAST
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from audiossl.utils.common import cosine_scheduler_step,get_params_groups
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch import nn
from audiossl.models.atst.byol import ByolLoss,build_mlp
import torch
from audiossl.methods.promptpool.audio_transformer import FrameAST_base, FrameAST_small


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

class ModelWrapper(nn.Module):

    def __init__(self, encoder,
                 embed_dim, 
                 train_mode,
                 predictor=True):
        super(ModelWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        self.encoder = encoder
        self.train_mode = train_mode
        self.projector = build_mlp(2,embed_dim,4096,256,last_bn=False)
        if predictor:
            self.predictor=build_mlp(2,256,4096,256,last_bn=False)
        else: 
            self.predictor=nn.Identity()

    def forward(self, x, length, query,mask_index=None,mask_input=False, avg=False):
        cls_repr,frm_repr = self.encoder(torch.cat(x),
                                        length=torch.cat(length),
                                        query=torch.cat(query),
                                        mask_index=None if mask_index is None else torch.cat(mask_index),
                                        mask_input=mask_input,
                                        avg=avg)

        if self.train_mode == "frame":
            return self.predictor(self.projector(frm_repr))
        elif self.train_mode == "cls":
            return self.predictor(self.projector(cls_repr))
        else:
            raise NotImplementedError






class PromptPool(nn.Module):
    def __init__(self,arch,train_mode,pool_size,prompt_len,select_num,prompt_key,pool):
        super().__init__()
        self.train_mode = train_mode
        encoder_fn=PromptPoolAST
        ast_fn = FrameAST_small if arch=="small" else FrameAST_base

        

        self.pool_size=pool_size
        self.pool=pool
        student=encoder_fn(ast_fn(),
                           pool_size=pool_size,
                           prompt_len=prompt_len,
                           select_num=select_num,
                           prompt_key=prompt_key,
                           pool=pool)
        self.embed_dim= student.framemodel.embed_dim if self.pool == "mean" else self.pool_size*student.framemodel.embed_dim 


        self.student=ModelWrapper(student,
                                  self.embed_dim,
                                  self.train_mode,
                                  predictor=True
                                      )
                
        self.teacher=ModelWrapper(encoder_fn(ast_fn(),
                                            pool_size=pool_size,
                                            prompt_len=prompt_len,
                                            select_num=select_num,
                                            prompt_key=prompt_key,
                                            pool=pool),
                                  self.embed_dim,
                                  self.train_mode,
                                  predictor=False
                                      )
        for p in self.teacher.parameters():
            p.requires_grad = False
        
        self._init_teacher()

        self.loss_fn = ByolLoss(2)

    def _init_teacher(self):
        self.teacher.load_state_dict({k:v for k,v in self.student.state_dict().items() if "predictor" not in k })

    def forward(self,x,length,query,mask_index=None):
        if self.train_mode=="cls":
            stu=self.student(x,length,query)
            tea=self.teacher(x,length,query)
        elif self.train_mode=="frame":
            stu=self.student(x,length,query,mask_index,True)
            tea=self.teacher(x,length,query,mask_index,False)
        else:
            raise NotImplementedError
        return self.loss_fn(stu,tea)

    def update_teacher(self,m):
        with torch.no_grad():
            for param_q, param_k in zip(self.student.encoder.parameters(), self.teacher.encoder.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
            for param_q, param_k in zip(self.student.projector.parameters(), self.teacher.projector.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)


class PromptPoolLightningModule(LightningModule):
    def __init__(self,
                 arch,
                 train_mode,
                 prompt_key,
                 pool_size=3,
                 prompt_len=2,
                 select_num=3,
                 pool="mean",
                 learning_rate:float=5e-4,
                 warmup_steps=1300,
                 max_steps=39000,
                 ema=0.99,
                 **kwargs,
                 ):
        super().__init__()
        self.model = PromptPool(arch,train_mode,pool_size,prompt_len,select_num,prompt_key,pool)
        self.train_mode = train_mode
        self.learning_rate = learning_rate 
        self.warmup_steps =  warmup_steps
        self.max_steps = max_steps
        self.ema_scheduler= cosine_scheduler_step(ema,1,max_steps,0)
        self.wd_scheduler = cosine_scheduler_step(0.04,0.4,max_steps,0)
        self.mylr_scheduler = cosine_scheduler_step(learning_rate,1e-6,max_steps,warmup_steps)
        self.save_hyperparameters()
    def training_step(self,batch,batch_idx):
        self.schedule()
        if self.train_mode == "cls":
            (melspecs,lengths),query,_ = batch
            byol_loss,std_s,std_t= self.model(melspecs,lengths,query)
        elif self.train_mode == "frame":
            (melspecs,lengths,masks),query,_ = batch
            byol_loss,std_s,std_t= self.model(melspecs,lengths,query,masks)
        else:
            raise NotImplementedError
        loss = byol_loss 
        self.log("loss",loss,prog_bar=True,logger=True)
        self.log("byol_loss",byol_loss,prog_bar=True,logger=True)
        self.log("std_t",std_t,prog_bar=True,logger=True)
        self.log("std_s",std_s,prog_bar=True,logger=True)
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
        params = get_params_groups(self.model.student)
        optimizer = AdamW(params,
                          lr=self.learning_rate,
                          weight_decay=0.)
        return [optimizer]
    def on_train_batch_end(self, outputs, batch, batch_idx: int, unused: int = 0) -> None:
        m = self.ema_scheduler[self.global_step]
        self.model.update_teacher(m)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("ClsPromptModel")
        parser.add_argument("--arch",type=str,default="small")
        parser.add_argument("--pool_size",type=int,default=12)
        parser.add_argument("--prompt_len",type=int,default=2)
        parser.add_argument("--select_num",type=int,default=3)
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