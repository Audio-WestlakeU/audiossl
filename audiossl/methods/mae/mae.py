
import torch
from torch import nn
from audiossl.models.atst.audio_transformer import PatchEmbed_v2,get_num_patches,trunc_normal_,Block
import random_mask
from torch.nn import functional as F


class Encoder(nn.Module):
    """ Vision Transformer """
    def __init__(self, spec_h=64,spec_w=1001, patch_w=16,patch_h=16, in_chans=1, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm,mask_ratio=0.5, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.spec_w = spec_w
        self.spec_h = spec_h
        self.embed_dim = embed_dim
        self.patch_w = patch_w
        self.patch_h = patch_h


        self.patch_embed = PatchEmbed_v2(patch_h,patch_w,embed_dim)

        num_patches = get_num_patches(spec_h,spec_w,patch_h,patch_w)
        self.num_patches = num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.mask_ratio = mask_ratio

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def prepare_tokens(self, x, mask=True):
        B, nc, h, w = x.shape
        mel_patches,x,patch_length = self.patch_embed(x)  # patch linear embedding
        B, T, C = x.shape
        mask_index = None

        if mask:
            mask_index = random_mask.get_mask_v2(B,T,self.mask_ratio).cuda()
            mask_index_expand = mask_index.unsqueeze(2).expand(B,T,self.embed_dim)

        # add positional encoding to each token
        pos = self.pos_embed[:,:T,:].expand(B,-1,-1) 
        x = x + pos

        return self.pos_drop(x),pos,mel_patches,mask_index,h,w

    def forward(self, x):
        x,pos,mel_patches,mask_index,h,w = self.prepare_tokens(x)
        B,T,C = x.shape
        x = x[~mask_index].reshape(B,-1,C)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        
        return x,mel_patches,mask_index,h,w

    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, n=1):
        x,_,_,_,_,_ = self.prepare_tokens(x,mask=False)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output

class Decoder(nn.Module):
    def __init__(self, patch_w=16, patch_h=16,  embed_dim=384, depth=6,
                 num_heads=6, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm
                 ):
        super().__init__()
        self.num_classes = patch_w * patch_h
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_w = patch_w
        self.patch_h = patch_h
        self.embed_dim = embed_dim

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, return_token_num):
        for blk in self.blocks:
            x = blk(x)

        if return_token_num > 0:
            x = self.head(self.norm(x[:, -return_token_num:])) # only return the mask tokens predict pixels
        else:
            x = self.head(self.norm(x)) # [B, N, 3*16^2]
        return x
def encoder_small(patch_h=16,patch_w=16,**kwargs):
    return Encoder(patch_h=patch_h,patch_w=patch_w,embed_dim=384,depth=12,num_heads=6,**kwargs)

def encoder_base():
    return Encoder(embed_dim=768,depth=12,num_heads=12)

class MaskedAutoEncoder(nn.Module):
    def __init__(self,patch_h=16,patch_w=16,**kwargs):
        super().__init__()
        self.encoder = encoder_small(patch_h=patch_h,patch_w=patch_w,**kwargs)
        self.decoder = Decoder(patch_h=patch_h,patch_w=patch_w)
        self.middle = nn.Linear(self.encoder.embed_dim,self.decoder.embed_dim)
        self.mask_embed = nn.Parameter(torch.zeros(1,1, self.decoder.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.encoder.num_patches + 1, self.decoder.embed_dim))
        trunc_normal_(self.mask_embed, std=.02)
        trunc_normal_(self.pos_embed, std=.02)



    def forward(self,x,run_decoder=True):
        enc_x,mel_patches,mask_index,h,w = self.encoder(x)
        if run_decoder:
            
            x = self.middle(enc_x)
            B,npatch,_ = mel_patches.shape
            C = x.shape[-1]

            x += self.pos_embed[~mask_index]
            x_mask = self.pos_embed[mask_index].reshape(B,-1,C) + self.mask_embed 
            num_mask = x_mask.shape[1]
            x = torch.cat([x,x_mask],dim=1)
            x = self.decoder(x,0)
            mel_mask = mel_patches[mask_index]
            x_mask = x[:,-num_mask:].reshape(B*num_mask,-1)
            mse_loss = F.mse_loss(mel_mask,x_mask)
            return enc_x[:,0,:],mse_loss
        else:
            return enc_x[:,0,:], torch.zeros([])

if __name__ =="__main__":
    mae = MaskedAutoEncoder()
    input = torch.randn(2,1,64,101)
    input=input.cuda()
    mae=mae.cuda()
    a,b=mae(input)
    c = mae(input,run_decoder=False)
