
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

    return 2 - 2 * (p * z).sum(dim=1).mean()
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



class ByolLoss(nn.Module):
    def __init__(self,ncrops):
        super().__init__()
        self.ncrops=ncrops
    def forward(self,student,teacher):
        std_cls_s=compute_var(F.normalize(student,dim=-1)).mean()
        std_cls_t=compute_var(F.normalize(teacher,dim=-1)).mean()
        student = student.chunk(self.ncrops)
        teacher = teacher.detach().chunk(2)
        total_loss = 0
        n_loss_terms = 0
        for iq,q in enumerate(teacher):
            for iv,v in enumerate(student):
                if iq==iv:
                    continue
            
                loss = byol_loss_func(q,v,simplified=False)
                n_loss_terms += 1
                total_loss += loss.mean()

        total_loss /= n_loss_terms
        return total_loss,std_cls_s,std_cls_t



class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """
    def __init__(self, encoder,
                 embed_dim, 
                 predictor=True):
        super(MultiCropWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        self.encoder = encoder
        self.projector = build_mlp(2,embed_dim,4096,256,last_bn=False)
        if predictor:
            self.predictor=build_mlp(2,256,4096,256,last_bn=False)
        else: 
            self.predictor=nn.Identity()

    def forward(self, x, length, avg=False):
        # convert to list
        if not isinstance(x, list):
            x = [x]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)
        start_idx, output = 0,torch.empty(0).to(x[0].device)

        for end_idx in idx_crops:
            _out = self.encoder(torch.cat(x[start_idx: end_idx]),
                                          length=torch.cat(length[start_idx:end_idx]),
                                          avg=avg)
            # accumulate outputs
            output = torch.cat((output, _out))
            start_idx = end_idx
        # Run the head forward on the concatenated features.
        return self.predictor(self.projector(output))


