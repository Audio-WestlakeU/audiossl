from audiossl.methods.promptpool.audio_transformer import FrameAST,trunc_normal_
import torch
from torch import nn
from audiossl.methods.promptpool.model import FrameATSTLightningModule

def get_pretraied_encoder(pretrained_ckpt_path):
    # get pretrained encoder

    s = torch.load(pretrained_ckpt_path)

    if 'pytorch-lightning_version' in s.keys():
        pretrained_model = FrameATSTLightningModule.load_from_checkpoint(
            pretrained_ckpt_path)
        pretrained_model.unfreeze()
        pretrained_encoder = pretrained_model.model.teacher.encoder
    return pretrained_encoder

class PromptPoolAST(nn.Module):
    def __init__(self,framemodel:FrameAST,pool="mean",nprompt=12,prompt_len=2,query_dim=768,select_num=3):
        super().__init__()
        self.framemodel = framemodel
        self.nprompt=nprompt
        self.pool=pool
        self.query_dim=query_dim
        self.prompt_pool = nn.Parameter(torch.zeros(nprompt, prompt_len, self.framemodel.embed_dim))
        self.prompt_keys = nn.Parameter(torch.zeros(nprompt,self.query_dim))
        self.select_num = select_num
        self.embed_dim=self.framemodel.embed_dim if self.pool=="mean" else self.nprompt*self.framemodel.embed_dim 
        trunc_normal_(self.prompts, std=.02)
    def forward(self,x,length,query,avg=None):
        x,_,_,_,_,patch_length=self.framemodel.prepare_tokens(x,mask_index=None,length=length,mask=False)
        B,T,C=x.shape
        prompts = self.prompt_select(query)
        nprompt = prompts.shape[0]
        x = torch.cat([prompts.expand(B,-1,-1),x],dim=1)
        for i,blk in enumerate(self.framemodel.blocks):
            x = blk(x,patch_length+nprompt)
        x=self.framemodel.norm_frame(x)
        if self.pool=="mean":
            return torch.mean(x[:,:nprompt],dim=1)
        else:
            return torch.reshape(x[:,:nprompt],(B*nprompt,C))


    def prompt_select(self,query): # query: 1*h
        top_keys = torch.topk(torch.nn.CosineSimilarity()(query,self.prompt_keys),self.select_num)
        prompts = self.prompt_pool[top_keys.indices]
        prompts = prompts.reshape(-1,prompts.shape[-1])
        return prompts


    def get_last_n_blocks(self,x,length, query, n,scene=True):
        x,_,_,_,_,patch_length=self.framemodel.prepare_tokens(x,mask_index=None,length=length,mask=False)
        B,T,C=x.shape
        x = torch.cat([self.prompts.expand(B,-1,-1),x],dim=1)
        prompts = self.prompt_select(query)
        nprompt = prompts.shape[0]
        x = torch.cat([prompts.expand(B,-1,-1),x],dim=1)
        output_cls=[]
        output_frame=[]
        for i,blk in enumerate(self.framemodel.blocks):
            x = blk(x,patch_length+nprompt)
            if i >= len(self.framemodel.blocks) - n:
                if self.pool=="mean":
                    output_cls.append(torch.mean(self.framemodel.norm_frame(x)[:,:nprompt],dim=1))
                else:
                    output_cls.append(torch.reshape(self.framemodel.norm_frame(x)[:,:nprompt],B*nprompt,C))
                if scene:
                    length_mask = torch.arange(x.shape[1]-nprompt).to(x.device) < patch_length.unsqueeze(1)
                    output_frame.append(torch.sum(self.framemodel.norm_frame(x)[:,nprompt:]*length_mask.unsqueeze(-1),dim=1)/(patch_length.unsqueeze(-1)+1e-6))
                else:
                    output_frame.append(self.framemodel.norm_frame(x)[:,nprompt:])
        return torch.cat(output_cls,dim=1),torch.cat(output_frame,dim=-1)

