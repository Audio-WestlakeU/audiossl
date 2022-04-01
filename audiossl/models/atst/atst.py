

from torch import nn
from audiossl.models.atst.byol import MultiCropWrapper,ByolLoss
from audiossl.models.atst.audio_transformer import AST_small,AST_base
import torch
class ATST(nn.Module):
    def __init__(self,arch="small",ncrops=2,**kwargs):
        super().__init__()
        if arch == "small":
            encoder_fn = AST_small
            embed_dim = 384
        elif arch == "base":
            encoder_fn = AST_base
            embed_dim = 768
        else:
            raise RuntimeError("arch {} is not implemented".format(arch))
        self.student=MultiCropWrapper(encoder_fn(**kwargs),embed_dim,predictor=True)
        self.teacher=MultiCropWrapper(encoder_fn(**kwargs),embed_dim,predictor=False)
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.load_state_dict({k:v for k,v in self.student.state_dict().items() if "predictor" not in k })
        self.loss_fn = ByolLoss(ncrops)
    def forward(self,melspecs,lengths):
        teacher_output = self.teacher(melspecs[:2],lengths[:2])  
        student_output = self.student(melspecs,lengths)
        loss = self.loss_fn(student_output,teacher_output)
        return loss
    def update_teacher(self,m):
        with torch.no_grad():
            for param_q, param_k in zip(self.student.encoder.parameters(), self.teacher.encoder.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
            for param_q, param_k in zip(self.student.projector.parameters(), self.teacher.projector.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        