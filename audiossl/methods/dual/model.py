from pytorch_lightning import LightningModule
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from audiossl.utils.common import cosine_scheduler_step,get_params_groups
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch import nn
from audiossl.methods.dual.dual import AST_small,AST_base
import torch
from torch.nn.functional import mse_loss
import torch.nn.functional as F

def build_expander(num_layers, input_dim, mlp_dim, output_dim):
    mlp = []
    for l in range(num_layers):
        dim1 = input_dim if l == 0 else mlp_dim
        dim2 = output_dim if l == num_layers - 1 else mlp_dim

        mlp.append(nn.Linear(dim1, dim2, bias=False))

        if l < num_layers - 1:
            mlp.append(nn.BatchNorm1d(dim2))
            mlp.append(nn.ReLU(inplace=True))

    return nn.Sequential(*mlp)

def variance_loss(z1: torch.Tensor) -> torch.Tensor:
    """Computes variance loss given batch of projected features z1 from view 1 and
    projected features z2 from view 2.
    Args:
        z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
        z2 (torch.Tensor): NxD Tensor containing projected features from view 2.
    Returns:
        torch.Tensor: variance regularization loss.
    """
    eps = 1e-4
    #z1 = F.normalize(z1,dim=-1)
    std_z1 = compute_var(z1)
    std_z1 = torch.sqrt(z1.var(dim=0)+eps)
    std_loss = torch.mean(F.relu(1 - std_z1)) 
    return std_loss,std_z1.mean()

def compute_var(y):
        y = y.view(-1, y.size(-1))
        zc = torch.tensor(y.size(0)).cuda()
        zs = y.sum(dim=0)
        zss = (y ** 2).sum(dim=0)

        torch.distributed.all_reduce(zc)
        torch.distributed.all_reduce(zs)
        torch.distributed.all_reduce(zss)

        var = zss / (zc - 1) - (zs ** 2) / (zc * (zc - 1))
        return torch.sqrt(var + 1e-6)


class DUAL(nn.Module):
    def __init__(self,arch):
        super().__init__()
        if arch == "small":
            self.patchnet = AST_small(use_cls=False,patch_h=16,patch_w=16)
            self.framenet = AST_small(use_cls=False,patch_h=64,patch_w=4)
        elif arch == "base":
            self.patchnet = AST_base(use_cls=False,patch_h=16,patch_w=16)
            self.framenet = AST_base(use_cls=False,patch_h=64,patch_w=4)
        self.patch_expander = build_expander(3,self.patchnet.embed_dim,8192,64*4)
        self.frame_expander = build_expander(3,self.framenet.embed_dim,8192,64*4)
        
    def forward(self,x_frame,x_patch,mask_frame,mask_patch):

        patch_x,patch_mel = self.patchnet(x_patch,mask_patch)
        frame_x,frame_mel = self.framenet(x_frame,mask_frame)

        mask = mask_patch | mask_frame

        patch_x = patch_x[mask]
        frame_x = frame_x[mask]

        patch_mel = patch_mel[mask]
        frame_mel = frame_mel[mask]


        T,C = patch_x.shape
        if 1: # average
            patch_x_big = patch_x.reshape(T//4,4,C)
            patch_x_big = torch.mean(patch_x_big,dim=1) 
            frame_x_big = frame_x.reshape(T//4,4,C)
            frame_x_big = torch.mean(frame_x_big,dim=1) 

        patch_x = self.patch_expander(patch_x)
        frame_x = self.frame_expander(frame_x)

        loss_mel_patch = mse_loss(patch_x,patch_mel)
        loss_mel_frame = mse_loss(frame_x,frame_mel)


        #p = F.normalize(patch_x, dim=-1)
        #z = F.normalize(frame_x, dim=-1)

        #loss =  2 - 2 * (p * z).sum(dim=1).mean()

        loss_dual = mse_loss(frame_x_big,patch_x_big)
        loss_uniform_patch,std_patch = variance_loss(patch_x_big.reshape(-1,patch_x_big.shape[-1]))
        loss_uniform_frame,std_frame = variance_loss(frame_x_big.reshape(-1,patch_x_big.shape[-1]))

        return loss_mel_patch,loss_mel_frame,loss_dual,loss_uniform_patch,loss_uniform_frame,std_patch,std_frame
    

class DUALLightningModule(LightningModule):
    def __init__(self,
                 arch="small",
                 learning_rate:float=5e-4,
                 warmup_steps=1300,
                 max_steps=39000,
                 **kwargs,
                 ):
        super().__init__()
        self.model = DUAL(arch=arch)
        self.learning_rate = learning_rate 
        self.warmup_steps =  warmup_steps
        self.max_steps = max_steps
        self.wd_scheduler = cosine_scheduler_step(0.05,0.05,max_steps,0)
        self.mylr_scheduler = cosine_scheduler_step(learning_rate,1e-6,max_steps,warmup_steps)
        self.save_hyperparameters()
    def training_step(self,batch,batch_idx):
        self.schedule()
        (melspecs_frame,melspecs_patch,lengths,masks_frame,masks_patch),_ = batch
        loss_mel_patch,loss_mel_frame,loss_dual,loss_uniform_patch,loss_uniform_frame,std_patch,std_frame = self.model(melspecs_frame,melspecs_patch,masks_frame,masks_patch)
        self.log("loss_mel_patch",loss_mel_patch,prog_bar=True,logger=True)
        self.log("loss_mel_frame",loss_mel_frame,prog_bar=True,logger=True)
        self.log("loss_dual",loss_dual,prog_bar=True,logger=True)
        self.log("loss_uniform_patch",loss_uniform_patch,prog_bar=True,logger=True)
        self.log("loss_uniform_frame",loss_uniform_frame,prog_bar=True,logger=True)
        self.log("std_patch",std_patch,prog_bar=True,logger=True)
        self.log("std_frame",std_frame,prog_bar=True,logger=True)
        self.log("step",self.global_step,prog_bar=True,logger=True)
        loss = loss_mel_patch + loss_mel_frame + loss_dual + loss_uniform_patch + loss_uniform_frame
        
        return loss
    def schedule(self):
        for i, param_group in enumerate(self.trainer.optimizers[0].param_groups):
            param_group["lr"] = self.mylr_scheduler[self.global_step]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = self.wd_scheduler[self.global_step]
        
        self.log("wd",self.wd_scheduler[self.global_step],prog_bar=True,logger=True)
        self.log("lr",param_group["lr"],prog_bar=True,logger=True)

    def configure_optimizers(self):
        optimizer = AdamW(get_params_groups(self.model),
                          lr=self.learning_rate,
                          weight_decay=0.)
        return [optimizer]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("DUALModel")
        parser.add_argument("--arch",type=str,default="small")
        parser.add_argument("--learning_rate", default=0.0005, type=float, help="""Learning rate at the end of
            linear warmup (highest LR used during training). The learning rate is linearly scaled
            with the batch size, and specified here for a reference batch size of 256.""")
        parser.add_argument('--warmup_steps',default=1300,type=int)
        parser.add_argument('--max_steps',default=39010,type=int)
        return parent_parser