# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
from torch.nn import functional as F

def build_mlp(num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
    mlp = []
    for l in range(num_layers):
        dim1 = input_dim if l == 0 else mlp_dim
        dim2 = output_dim if l == num_layers - 1 else mlp_dim

        mlp.append(nn.Linear(dim1, dim2, bias=False))

        if l < num_layers - 1:
            mlp.append(nn.BatchNorm1d(dim2))
            mlp.append(nn.ReLU(inplace=True))
        elif last_bn:
            # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
            # for simplicity, we further removed gamma in BN
            mlp.append(nn.BatchNorm1d(dim2, affine=False))

    return nn.Sequential(*mlp)
def build_mlp2(num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
    mlp = []
    for l in range(num_layers):
        dim1 = input_dim if l == 0 else mlp_dim
        dim2 = output_dim if l == num_layers - 1 else mlp_dim

        mlp.append(nn.Linear(dim1, dim2, bias=False))

        if l < num_layers - 1:
            mlp.append(nn.LayerNorm(dim2))
            mlp.append(nn.ReLU(inplace=True))
        elif last_bn:
            # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
            # for simplicity, we further removed gamma in BN
            mlp.append(nn.BatchNorm1d(dim2, affine=False))

    return nn.Sequential(*mlp)

def byol_loss_func_cls(p: torch.Tensor, z: torch.Tensor, simplified: bool = True) :
    """
    Computes BYOL's loss given batch of predicted features p and projected momentum features z.
    Args:
        p (torch.Tensor): NxD Tensor containing predicted features from view 1
        z (torch.Tensor): NxD Tensor containing projected momentum features from view 2
        simplified (bool): faster computation, but with same result. Defaults to True.
    Returns:
        torch.Tensor: BYOL loss.
    """

    if simplified:
        return 2 - 2 * F.cosine_similarity(p, z, dim=-1).mean()

    p = F.normalize(p, dim=-1)
    z = F.normalize(z, dim=-1)

    return 2 - 2 * (p * z).sum(dim=1).mean()


def byol_loss_func(p: torch.Tensor, z: torch.Tensor, simplified: bool = True) :
    """
    Computes BYOL's loss given batch of predicted features p and projected momentum features z.
    Args:
        p (torch.Tensor): NxD Tensor containing predicted features from view 1
        z (torch.Tensor): NxD Tensor containing projected momentum features from view 2
        simplified (bool): faster computation, but with same result. Defaults to True.
    Returns:
        torch.Tensor: BYOL loss.
    """

    if simplified:
        return 2 - 2 * F.cosine_similarity(p, z, dim=-1).mean()

    p = F.normalize(p, dim=-1)
    z = F.normalize(z, dim=-1)

    return (2 - 2 * (p * z).sum(dim=1)).sum()

def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

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
    z1 = F.normalize(z1,dim=-1)
    std_z1 = compute_var(z1)
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



def covariance_loss(z1: torch.Tensor) -> torch.Tensor:
    """Computes covariance loss given batch of projected features z1 from view 1 and
    projected features z2 from view 2.
    Args:
        z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
        z2 (torch.Tensor): NxD Tensor containing projected features from view 2.
    Returns:
        torch.Tensor: covariance regularization loss.
    """

    N, D = z1.size()

    z1 = z1 - z1.mean(dim=0)
    cov_z1 = (z1.T @ z1) / (N - 1)

    diag = torch.eye(D, device=z1.device)
    cov_loss = cov_z1[~diag.bool()].pow_(2).sum() / D 
    return cov_loss

class ByolLoss(nn.Module):
    def __init__(self,ncrops,use_cls):
        super().__init__()
        self.ncrops = ncrops
        self.use_cls = use_cls
    def forward(self,student,teacher):

        q1,q2 = student
        k1,k2 = teacher
        B = k1.shape[0]
        loss_uniform_frm,std_frm = variance_loss(q2) 
        loss_uniform_cls,std_cls = variance_loss(q1) 
        _,std_frm_t = variance_loss(k2) 
        _,std_cls_t = variance_loss(k1) 


        loss_cls=torch.zeros([]).cuda()
        n_loss_cls=0
            #loss_cls += byol_loss_func(q1,k1)
            #n_loss_cls += 1
        if self.ncrops > 1:
            q1=q1.chunk(self.ncrops)
            k1=k1.chunk(1)
            for i in range(0,self.ncrops):
                if i==0:
                    continue
                loss_cls += byol_loss_func(q1[i],k1[0],simplified=False)/B
                n_loss_cls+=1

        #if self.use_cls or self.ncrops>1:
        #    loss_cls /= n_loss_cls

        q2 = q2
        k2 = k2

        loss_align = byol_loss_func(q2,k2,simplified=False)/B
        #loss_uniform= byol_loss_func(qv1,kv1,simplified=False)
            
        return loss_cls,loss_align,loss_uniform_frm,std_frm,loss_uniform_cls,std_cls,std_frm_t,std_cls_t

        student = student.chunk(self.ncrops)
        teacher = teacher.detach().chunk(1)

        total_loss = 0
        n_loss_terms = 0
        for iq,q in enumerate(teacher):
            for iv,v in enumerate(student):
                if iq==iv:
                    loss = byol_loss_func(q,v,simplified=False)
                    n_loss_terms += 1
                    total_loss += loss.mean()

        total_loss /= n_loss_terms
        return total_loss



class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """
    def __init__(self, backbone,embed_dim,nonlinear=False, predictor=False,use_cls=True,use_bn=False):
        super(MultiCropWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        self.use_bn = use_bn
        self.use_cls = use_cls

        if nonlinear:

            if self.use_bn:
                self.head = build_mlp(2,embed_dim,4096,256,last_bn=False)
            else:
                self.head = build_mlp2(2,embed_dim,4096,256,last_bn=False)
            if use_cls:
                self.head2 = build_mlp(2,embed_dim,4096,256,last_bn=False)

            if predictor:
                if self.use_bn:
                    self.predictor=build_mlp(2,256,4096,256,last_bn=False)
                else:
                    self.predictor=build_mlp2(2,256,4096,256,last_bn=False)
                if use_cls:
                    self.predictor2=build_mlp(2,256,4096,256,last_bn=False)
            else: 
                self.predictor=nn.Identity()
                if use_cls:
                    self.predictor2=nn.Identity()
        else:
            self.head=nn.Identity()
            self.head2=nn.Identity()
            if predictor:
                self.predictor=nn.Linear(embed_dim,embed_dim)
            else:
                self.predictor=nn.Identity()

    def forward(self, x, mask, length,src_idx, use_mask,avg, use_aug_loss ):
        # convert to list
        if not isinstance(x, list):
            x = [x]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)
        start_idx, output_cls,output_frm = 0, torch.empty(0).to(x[0].device), torch.empty(0).to(x[0].device)

        for end_idx in idx_crops:
            _out_cls, _out_frm = self.backbone(torch.cat(x[start_idx: end_idx]),
                                                          torch.cat(mask[start_idx:end_idx]),
                                                          torch.cat(length[start_idx:end_idx]),
                                                          torch.cat(src_idx[start_idx:end_idx]) if src_idx is not None else None,
                                                          use_mask,
                                                          avg,
                                                          use_aug_loss)
            # The output is a tuple with XCiT model. See:
            # https://github.com/facebookresearch/xcit/blob/master/xcit.py#L404-L405
            if isinstance(_out_cls, tuple):
                _out = _out[0]
            # accumulate outputs
            output_cls = torch.cat((output_cls, _out_cls))
            output_frm = torch.cat((output_frm, _out_frm))
            start_idx = end_idx
        # Run the head forward on the concatenated features.
        if self.use_cls:

            return self.predictor2(self.head2(output_cls)),self.predictor(self.head(output_frm))
        else:
            return output_cls,self.predictor(self.head(output_frm))

class Byol(nn.Module):
    """
    Build a Byol model with: a query encoder, a key encoder, and a queue
    """
    def __init__(self, base_encoder, num_classes=256,  K=65536, m=0.999, T=0.07):
        """
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(Byol, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder()
        self.encoder_k = base_encoder()
        dim=self.encoder_q.embed_dim

        self.head_q = self._build_mlp(2,dim,4096,num_classes,last_bn=False)
        self.head_k = self._build_mlp(2,dim,4096,num_classes,last_bn=False)
        self.predictor_q = self._build_mlp(2,num_classes,4096,num_classes,last_bn=False)


        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        for param_q, param_k in zip(self.head_q.parameters(), self.head_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient


    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_q, param_k in zip(self.head_q.parameters(), self.head_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q,len_q,mask_q, im_k,len_k,mask_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q,loss_mse_q = self.encoder_q(im_q,mask_q,len_q)  # queries: NxC
        q = self.head_q(q)
        q = self.predictor_q(q)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k,_ = self.encoder_k(im_k,mask_k,len_k)  # keys: NxC
            k = self.head_k(k)
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        loss_byol = byol_loss_func(q,k)

        return loss_byol, loss_mse


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
