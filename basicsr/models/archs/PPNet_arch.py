""" Vision Transformer (ViT) in PyTorch

A PyTorch implement of Vision Transformers as described in
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale' - https://arxiv.org/abs/2010.11929

The official jax code is released and available at https://github.com/google-research/vision_transformer

Status/TODO:
* Models updated to be compatible with official impl. Args added to support backward compat for old PyTorch weights.
* Weights ported from official jax impl for 384x384 base and small models, 16x16 and 32x32 patches.
* Trained (supervised on ImageNet-1k) my custom 'small' patch model to 77.9, 'base' to 79.4 top-1 with this code.
* Hopefully find time and GPUs for SSL or unsupervised pretraining on OpenImages w/ ImageNet fine-tune in future.

Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert

Hacked together by / Copyright 2020 Ross Wightman
"""
from tkinter import X
import torch
import torch.nn as nn
from thop import profile
from timm.models.layers import DropPath
import torch.nn.functional as F
from models.archs.arch_util import LayerNorm,Mlp
#####Global Convolutional Module
class GCM(nn.Module):
    def __init__(self,in_dim,out_dim,  attn_drop=0.,kernel_size=4,stride=4,padding=0,groups=1,
                 drop_path=0., norm_layer=LayerNorm,bias=False):
        super(GCM, self).__init__()
        #out_dim=out_dim*stride
        self.down=nn.Sequential(nn.Conv2d(in_dim,out_dim,kernel_size=1,stride=1, padding=0, bias=bias),
                                nn.UpsamplingBilinear2d(scale_factor=1.0/stride))
        mlp_ratio=2
        self.norm1=norm_layer(in_dim)
        self.block=nn.Sequential(nn.Conv2d(in_dim,out_dim,kernel_size=1,stride=1,padding=0,groups=1,bias=bias),
                                nn.Conv2d(out_dim,out_dim,kernel_size=kernel_size,stride=stride,padding=(kernel_size//2),groups=out_dim,bias=bias))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.reweight= Mlp(out_dim, out_dim // 4, out_dim*2)

        self.norm2=norm_layer(out_dim)
        mlp_hidden_dim = int(out_dim * mlp_ratio)
        self.mlp = Mlp(in_features=out_dim, hidden_features=mlp_hidden_dim)
        
    def forward(self, x):
        x=self.norm1(x)
        d_x=self.down(x)
        x=self.drop_path(self.block(x))
        f=x+d_x
        B,C,H,W=f.shape
        f=F.adaptive_avg_pool2d(f,output_size=1)
        f=self.reweight(f).reshape(B, C, 2).permute(2, 0, 1).softmax(dim=0).unsqueeze(-1).unsqueeze(-1)
        x=f[0]*x+f[1]*d_x
        x=x+self.mlp(self.norm2(x))
        del d_x,f
        return x

class CrossAttention(nn.Module):
    r""" Multi-head self attention module with dynamic position bias.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, q_dim,kv_dim,patches,attn_mode='patch_wise', qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,):

        super().__init__()
        self.attn_mode=attn_mode
        self.patches=patches
        self.patch_h=patches[0]
        self.patch_w=patches[1]

        self.proj_q=nn.Conv2d(q_dim,patches[0]*patches[1],1,1,0,bias=qkv_bias)
        self.proj_k=nn.Conv2d(kv_dim,patches[0]*patches[1],1,1,0,bias=qkv_bias)
        self.proj_v=nn.Conv2d(kv_dim,patches[0]*patches[1],1,1,0,bias=qkv_bias)

        self.c_k = CrossPatchModule(patches=patches,mode='same')
        self.c_v = CrossPatchModule(patches=patches,mode='same')

        self.attn_drop = nn.Dropout(attn_drop)  
        self.proj = nn.Conv2d(patches[0]*patches[1],kv_dim,1,1,0)
        self.proj_drop = nn.Dropout(proj_drop)

    def calculate_cross_attn(self,q,k,v):
        
        if self.attn_mode=='pixel_wise':
            #print('pixel_wise')
            k = k.flatten(2) # [B,patch_n,P_h*P_w]
            #expand q channels = P_h*P_w
            r=k.shape[2]/q.shape[2]
            q=q.repeat((1,1,int(r))) #[B,patch_n,P_h*P_w]
            attn= (q.transpose(1,2) @ k) 
            del q,k
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn) #[B,P_h*P_w,P_h*P_w]
            v = v.flatten(2) # [B,P_h*P_w,patch_n]
            attn=(v@attn.transpose(1,2))# [B,P_h*P_w,patch_n]
            del v
            return attn # [B,patch_n,P_h*P_w]

        if self.attn_mode=='patch_wise':
            #print('patch_wise')
            k=k.flatten(2).transpose(1,2) # [B,P_h*P_w,patch_n]
            #expand q channels = P_h*P_w
            r=k.shape[1]/q.shape[2]
            q=q.repeat((1,1,int(r))) #[B,patch_n,P_h*P_w]
            attn=(q@k)
            del q,k
            attn=attn.softmax(dim=-1)
            attn=self.attn_drop(attn) #[B,patch_n,patch_n]

            v = v.flatten(2) # [B,P_h*P_w,patch_n]
            attn=(attn@v)  # [B,P_h*P_w,patch_n]
            del v
            return attn #[B,patch_n,P_h*P_w]


    def attn_reshape(self,x):
        B,C,patch_n,kernel_h,kernel_w=x.shape
        H=self.patch_h*kernel_h
        W=self.patch_w*kernel_w

        #x=x.view(B,-1,patch_n, kernel_h, kernel_w) # [B,C,patch_num,kernel_h,kernel_w]
        x=x.permute(0,1,3,4,2).view(B,-1,kernel_h*kernel_w,self.patch_h*self.patch_w)
        x=x.reshape(B,-1,self.patch_h*self.patch_w)
        fold=torch.nn.Fold(output_size=(H,W),kernel_size=(kernel_h,kernel_w),stride=(kernel_h,kernel_w))
        x=fold(x)
        #print(x.shape)
        return x #[B,1,H,W]
    def forward(self,q,lf):
        """
        Args:
            lf:local features
            q: input features with shape of (num_groups*B,C,P_h,P_w)
            mask: (0/-inf) mask with shape of (num_groups, Gh*Gw, Gh*Gw) or None
            H: height of each group
            W: width of each group
        """
        B,C,H,W=lf.shape
        attn_i=[]
        #B_, C_,P_h,P_w = q.shape
        q=self.proj_q(q)
        q = q.flatten(2) #[B,C,patches_h*patches_w]
        
        k=self.proj_k(lf)
        v=self.proj_v(lf)
        del lf
        #reshape local feature maps
        k=self.c_k(k)# [B,C,patch_n,kernel_h,kernel_w]
        v=self.c_v(v)# [B,C,patch_n,kernel_h,kernel_w]


        kernel_h=H//self.patch_h
        kernel_w=W//self.patch_w
        #calculate cross attention
        for i in range(q.shape[1]):
            attn=self.calculate_cross_attn(q,k[:,i,:,:],v[:,i,:,:])#[B,patch_n,kernel_h,kernel_w]
            attn=attn.view(B,q.shape[1],kernel_h,kernel_w).unsqueeze(1)#[B,1,patch_n,kernel_h,kernel_w]
            attn_i.append(attn)
            del attn
        del q,k,v
        out=torch.cat(attn_i,dim=1)
        del attn_i
        out=self.attn_reshape(out)
        out = self.proj(out)
        out = self.proj_drop(out)

        return out

class TransformerBlock(nn.Module):
    r"""  Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self,
                 patches,
                 q_dim,
                 kv_dim,
                 attn_mode='patch_wise',
                 mlp_ratio=2.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.PReLU,
                 norm_layer=LayerNorm):
        super().__init__()
        self.mlp_ratio = mlp_ratio

        self.norm_x = norm_layer(q_dim)
        self.norm_y = norm_layer(kv_dim)

        self.attn = CrossAttention(
            q_dim,kv_dim,patches=patches,attn_mode=attn_mode,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm = norm_layer(kv_dim)
        mlp_hidden_dim = int(kv_dim * mlp_ratio)
        self.mlp = Mlp(in_features=kv_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, y):

        # FFN
        y = y + self.drop_path(self.attn(self.norm_x(x),self.norm_y(y)))
        y = y + self.drop_path(self.mlp(self.norm(y)))
        #return self.attn(self.norm_x(x),self.norm_y(y))
        
        return y


class FFB(nn.Module):
    def __init__(self, in_dim,out_dim,bias=False,proj_drop=0.,norm=LayerNorm):
        super().__init__()
 
        self.fc=nn.Conv2d(in_dim,out_dim,1,1,0,bias=bias)
        self.gate=nn.Sequential(nn.Conv2d(in_dim,out_dim,1,1,0,bias=bias),
                                nn.Tanh(),
                                nn.Conv2d(out_dim,out_dim,kernel_size=7,stride=1,padding=7//2,groups=out_dim,bias=bias)
                                )
        self.wise_features=nn.Sequential(nn.Conv2d(in_dim,out_dim,1,1,0,bias=bias),
                                nn.PReLU(),
                                nn.Conv2d(out_dim,out_dim,kernel_size=7,stride=1,padding=7//2,groups=out_dim,bias=bias)
                                )

        self.reweight = Mlp(out_dim, out_dim // 4, out_dim * 3)
        self.proj = nn.Conv2d(out_dim, out_dim, 1, 1,bias=True)
        self.proj_drop = nn.Dropout(proj_drop)   
 
    def forward(self, x):
     
        x_fc=self.fc(x)
        x_wf=self.wise_features(x)
        x_gate=self.gate(x)
        f=x_fc+x_wf+x_gate

        B, C, H, W = f.shape
        f=F.adaptive_avg_pool2d(f,output_size=1)
        f=self.reweight(f).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(-1).unsqueeze(-1)
        x=f[0]*x_fc+f[1]*x_wf+f[2]*x_gate

        del x_fc,x_wf,x_gate
        x = self.proj(x)
        x = self.proj_drop(x)           
        return x

class FEB(nn.Module):

    def __init__(self, norm_dim,dim, mlp_ratio=2., bias=False,  attn_drop=0.,
                 drop_path=0., act_layer=nn.PReLU, norm_layer=LayerNorm,):
        super().__init__()
        self.norm1=norm_layer(norm_dim)
        self.conv=nn.Conv2d(norm_dim,dim,1,1,0,bias=bias)
        self.block = FFB(norm_dim,dim,bias=bias,norm=norm_layer)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        
    def forward(self, x):
        x = self.conv(x)+ self.drop_path(self.block(self.norm1(x))) 
        x = x + self.drop_path(self.mlp(self.norm2(x))) 
        return x

def basic_blocks(embed_dims, index, layers,mlp_ratio=3., bias=False,  attn_drop=0.,
                 drop_path_rate=0.,norm_layer=LayerNorm, **kwargs):
    blocks = []
    norm_index=0
    idx=index+1
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
        norm_index=idx-1
        blocks.append(FEB(embed_dims[norm_index],embed_dims[idx], mlp_ratio=mlp_ratio, bias=bias, 
                      attn_drop=attn_drop, drop_path=block_dpr, norm_layer=norm_layer,))
    blocks = nn.Sequential(*blocks)
    return blocks 
class CrossPatchModule(nn.Module):
    def __init__(self, patches=[8,8],mode='cross',):
        super().__init__()
        self.mode=mode
        self.patch_h=patches[0]
        self.patch_w=patches[1]
        self.patch_n=self.patch_h*self.patch_w
        self.step=self.patch_n//self.patch_n
        #pos embed
        absolute_pos_embed = nn.Parameter(torch.zeros(1,self.step, self.patch_n, self.patch_n,1,1))
        self.abs_pos=nn.init.trunc_normal_(absolute_pos_embed, std=.02)

    def forward(self,x):
        B,C,H,W=x.shape

        kernel_h=H//self.patch_h
        kernel_w=W//self.patch_w
        ###slicing and recombination
        if self.mode=='cross':
            #print('cross')
            unfold=torch.nn.Unfold(kernel_size=(kernel_h,kernel_w),stride=(kernel_h,kernel_w))
            x=unfold(x) #[N, C*kernel_size*kernel_size, self.patch_h*self.patch_w]
            x=x.view(B,-1,kernel_h,kernel_w,self.patch_h,self.patch_w) #[N, C, kernel_size, kernel_size, patch_h, patch_w]
            x=x.view(B,-1,kernel_h,kernel_w,self.patch_h*self.patch_w).permute(0,1,4,2,3)#[N, C, patch_h*patch_w(patch_n), kernel_size, kernel_size]
            x=x.view(B, self.step, x.shape[1]//self.step, self.patch_n, kernel_h, kernel_w)
            x=x+self.abs_pos

            #rolling_cross_channel_cross_patch
            for st in range(self.step): #step means groups
                for m in range(self.patch_n):
                    idx_i=[] # changing index list
                    for i in range(m,self.patch_n):
                        idx_i.append(i) #eg. (6,64)
                    if m>0:
                        for j in range(m): 
                            idx_i.append(j) #eg (0,6) idx_i=[6,7,8,...,64,0,1,2,3,4,5]
                    x[:,st,m,:]=x[:,st,m,idx_i] #change patches order

            x=x.view(B,-1,self.patch_n, kernel_h, kernel_w) # [B,C,patch_num,kernel_h,kernel_w]
            x=x.permute(0,1,3,4,2).view(B,-1,kernel_h*kernel_w,self.patch_h*self.patch_w)
            x=x.view(B,-1,self.patch_h*self.patch_w)
            fold=torch.nn.Fold(output_size=(H,W),kernel_size=(kernel_h,kernel_w),stride=(kernel_h,kernel_w))
            x=fold(x)
            return x

        elif self.mode=='same':
            #print('same')
            #same_patch_same_channel
            
            unfold=torch.nn.Unfold(kernel_size=(kernel_h,kernel_w),stride=(kernel_h,kernel_w))
            x=unfold(x) #[N, C*kernel_size*kernel_size, self.patch_h*self.patch_w]
            x=x.view(B,-1,kernel_h,kernel_w,self.patch_h,self.patch_w) #[N, C, kernel_size, kernel_size, patch_h, patch_w]
            x=x.view(B,-1,kernel_h,kernel_w,self.patch_h*self.patch_w).permute(0,1,4,2,3)#[N, C, patch_h*patch_w(patch_n), kernel_size, kernel_size]
            x=x.view(B, self.step, x.shape[1]//self.step, self.patch_n, kernel_h, kernel_w)

            x=x+self.abs_pos
            return x.reshape(B,-1,self.patch_n,kernel_h,kernel_w)
        else:
            raise ValueError(f'please enter the mode: same or cross')



       
class PPNet(nn.Module):
    def __init__(self,in_channel=3,local_embed_dims=[16,32,32,32,16],layers=[1,1,1,1],mlp_ratios=[1,1,1,1],
        bias=False, attn_drop_rate=0., drop_path_rate=0.,patches=[8,8],global_embed_dims=[16,32,32,32,32,16],gcm_ks=[[7,2],[7,2],[7,2],[7,2],[7,2]],
        norm_layer=LayerNorm):
        super().__init__()

        local_branch=[]
        global_branch=[]

        self.local_shallow_feats=nn.Conv2d(in_channel,local_embed_dims[0],kernel_size=3,stride=1,padding=1,bias=False)

        self.global_shallow_feats=nn.Conv2d(in_channel,patches[0]*patches[1],kernel_size=3,stride=1,padding=1,bias=False)
        self.cpm=CrossPatchModule(patches=patches,mode='cross')
        self.global_cfc=nn.Conv2d(patches[0]*patches[1],global_embed_dims[0],kernel_size=1,stride=1,padding=0)

        for i in range(len(layers)):
            local_layer = basic_blocks(local_embed_dims, i,layers,mlp_ratio=mlp_ratios[i], bias=bias,
                                 attn_drop=attn_drop_rate, drop_path_rate=drop_path_rate,
                                 norm_layer=norm_layer)
            local_branch.append(local_layer)
            if i >= len(layers) - 1:
                break
        self.local_branch = nn.ModuleList(local_branch)

        for i in range(len(global_embed_dims)-1):
            layer = GCM(global_embed_dims[i],global_embed_dims[i+1],kernel_size=gcm_ks[i][0],stride=gcm_ks[i][1])
            global_branch.append(layer)
            if i >= len(global_embed_dims) - 2:
                break

        self.global_branch = nn.ModuleList(global_branch)

        self.pixel_wise_attn=TransformerBlock(patches,global_embed_dims[-1],local_embed_dims[-1],attn_mode='pixel_wise')
        self.patch_wise_attn=TransformerBlock(patches,global_embed_dims[-1],local_embed_dims[-1],attn_mode='patch_wise')
        # #delete block effect
        self.pixel_attn_fusion=nn.Sequential(nn.Conv2d(local_embed_dims[-1],local_embed_dims[-1],1,1,0,bias=bias),
                                nn.PReLU(),
                                nn.Conv2d(local_embed_dims[-1],local_embed_dims[-1],kernel_size=7,stride=1,padding=7//2,groups=local_embed_dims[-1],bias=bias)
                                )
        self.patch_attn_fusion=nn.Sequential(nn.Conv2d(local_embed_dims[-1],local_embed_dims[-1],1,1,0,bias=bias),
                                nn.PReLU(),
                                nn.Conv2d(local_embed_dims[-1],local_embed_dims[-1],kernel_size=7,stride=1,padding=7//2,groups=local_embed_dims[-1],bias=bias)
                                )

        self.ch_attn=Mlp(local_embed_dims[-1], local_embed_dims[-1] // 4, local_embed_dims[-1]*2)
        self.out=nn.Sequential(nn.Conv2d(local_embed_dims[-1],local_embed_dims[-1]*2,3,1,1,bias=False),nn.PReLU(),
                                nn.Conv2d(local_embed_dims[-1]*2,3,3,1,1,bias=False))



    def forward(self,x):
        #compute local_branch
        local_shallow_feats=self.local_shallow_feats(x)

        for idx,block in enumerate(self.local_branch):
            local_shallow_feats=block(local_shallow_feats)

        #compute global_branch

        global_shallow_feats=self.global_shallow_feats(x)
        global_shallow_feats=self.cpm(global_shallow_feats)
        global_shallow_feats= self.global_cfc(global_shallow_feats)

        for idx,block in enumerate(self.global_branch):
            global_shallow_feats=block(global_shallow_feats)
        #calculate correaltion between global and local
        #q:global v:local
        pixel_attn=self.pixel_wise_attn(global_shallow_feats,local_shallow_feats)
        patch_attn=self.patch_wise_attn(global_shallow_feats,local_shallow_feats)
        del global_shallow_feats,local_shallow_feats

        pixel_attn=self.pixel_attn_fusion(pixel_attn)        
        patch_attn=self.patch_attn_fusion(patch_attn)
        #blocking artifacts
        deep_fusion_feats=pixel_attn+patch_attn
        
        B,C,H,W=deep_fusion_feats.shape
        deep_fusion_feats=F.adaptive_avg_pool2d(deep_fusion_feats,output_size=1)
        deep_fusion_feats=self.ch_attn(deep_fusion_feats).reshape(B, C, 2).permute(2, 0, 1).softmax(dim=0).unsqueeze(-1).unsqueeze(-1)
        deep_fusion_feats=deep_fusion_feats[0]*pixel_attn+deep_fusion_feats[1]*patch_attn
        del pixel_attn,patch_attn
        deep_fusion_feats=self.out(deep_fusion_feats)

        return  x+deep_fusion_feats

if __name__=='__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=PPNet()
    model = model.to(device)
    if str(device) =='cuda':
        input=torch.randn(1,3,256,256).cuda()
    else:
        input=torch.randn(1,3,256,256)
    print(model)
    flops,params=profile(model,inputs=(input,))
    print('flops:{}G params:{}M'.format(flops/1e9,params/1e6))