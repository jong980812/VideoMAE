# some codes from CLIP github(https://github.com/openai/CLIP), from VideoMAE github(https://github.com/MCG-NJU/VideoMAE)
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from collections import OrderedDict
from einops import rearrange
import clip_models.clip as clip


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 400, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)
    
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, num_frames=16, tubelet_size=2):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.tubelet_size = int(tubelet_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (num_frames // self.tubelet_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv3d(in_channels=in_chans, out_channels=embed_dim, 
                            kernel_size = (self.tubelet_size,  patch_size[0],patch_size[1]), 
                            stride=(self.tubelet_size,  patch_size[0],  patch_size[1]))

    def forward(self, x, **kwargs):
        B, C, T, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
    
# sin-cos position encoding
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31
def get_sinusoid_encoding_table(n_position, d_hid): 
    ''' Sinusoid position encoding table ''' 
    # TODO: make it with torch instead of numpy 
    def get_position_angle_vec(position): 
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)]) 
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 

    return torch.FloatTensor(sinusoid_table).unsqueeze(0) 

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement 
        x = self.fc2(x)
        x = self.drop(x)
        return x

# 기존 weight load편의성을 위해 Attention이름을 유지한다.
class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        s2t_q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        s2t_q = s2t_q * self.scale
        attn = (s2t_q @ k.transpose(-2, -1))

        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
# spatial to temporal cross attention module.
class CrossAttentionS2T(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        #self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        # cross attn split weight
        self.s2t_q = nn.Linear(dim, all_head_dim, bias=False)
        self.s2t_kv = nn.Linear(dim, all_head_dim * 2, bias=False)
        if qkv_bias:
            self.s2t_q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.s2t_kv_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.s2t_q_bias = None
            self.s2t_kv_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, s_x, t_x):
        B, s_N, C = s_x.shape
        _, t_N, C = t_x.shape
        s2t_q_bias = None
        s2t_kv_bias = None
        if self.s2t_q_bias is not None:
            s2t_q_bias = self.s2t_q_bias
            s2t_kv_bias = torch.cat((torch.zeros_like(self.s2t_kv_bias, requires_grad=False), self.s2t_kv_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        
        # querry = temporal input
        s2t_q = F.linear(input=t_x, weight=self.s2t_q.weight, bias=s2t_q_bias)
        s2t_q = s2t_q.reshape(B, t_N, self.num_heads, -1).permute(0, 2, 1, 3)
        # s2t_kv = spatial input
        s2t_kv = F.linear(input=s_x, weight=self.s2t_kv.weight, bias=s2t_kv_bias)
        s2t_kv = s2t_kv.reshape(B, s_N, 2, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        s2t_q, s2t_k, s2t_v = s2t_q, s2t_kv[0], s2t_kv[1]   # make torchscript happy (cannot use tensor as tuple)

        s2t_q = s2t_q * self.scale
        s2t_attn = (s2t_q @ s2t_k.transpose(-2, -1))

        
        s2t_attn = s2t_attn.softmax(dim=-1)
        s2t_attn = self.attn_drop(s2t_attn)

        t_x = (s2t_attn @ s2t_v).transpose(1, 2).reshape(B, t_N, -1)
        t_x = self.proj(t_x)
        t_x = self.proj_drop(t_x)
        return t_x


# this codes from CLIP github(https://github.com/openai/CLIP)
class CrossAttentionT2S(nn.Module): # 이게 VMAE로 치면 blocks class다. 여기에 cross s2t_attn layer가 추가되어야 한다.
    def __init__(self, dim: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        # add for cross-attn
        self.num_head = n_head
        head_dim = dim // self.num_head
        self.scale = head_dim ** -0.5
        all_head_dim = head_dim * self.num_head
        
        #여기에 cross attn t2s module이 들어가야 한다.
        self.t2s_q = nn.Linear(dim, all_head_dim, bias=False)
        self.t2s_q_bias = nn.Parameter(torch.zeros(dim))
        
        self.t2s_kv = nn.Linear(dim, all_head_dim * 2, bias=False)
        self.t2s_kv_bias = nn.Parameter(torch.zeros(dim))
        
        self.t2s_proj = nn.Linear(all_head_dim, dim)
        # 여기에 drop out 할지 말지는 고민좀 해보자.
        
        self.attn_mask = attn_mask
    
    def t2s_cross_attn(self, s_x, t_x):
        B, s_N, C = s_x.shape
        _, t_N, C = t_x.shape
        t2s_q_bias = self.t2s_q_bias
        t2s_kv_bias = torch.cat((torch.zeros_like(self.t2s_kv_bias, requires_grad=False), self.t2s_kv_bias))
        
        t2s_q = F.linear(input=s_x, weight=self.t2s_q.weight, bias=t2s_q_bias)
        t2s_q = t2s_q.reshape(B, s_N, self.num_head, -1).permute(0, 2, 1, 3)
        t2s_kv = F.linear(input=t_x, weight=self.t2s_kv.weight, bias=t2s_kv_bias)
        t2s_kv = t2s_kv.reshape(B, t_N, 2, self.num_head, -1).permute(2, 0, 3, 1, 4)
        t2s_q, t2s_k, t2s_v = t2s_q, t2s_kv[0], t2s_kv[1]
        
        t2s_q = t2s_q * self.scale
        t2s_attn = (t2s_q @ t2s_k.transpose(-2, -1))
        
        t2s_attn = t2s_attn.softmax(dim=-1)
        
        s_x = (t2s_attn @ t2s_v).transpose(1, 2).reshape(B, s_N, -1)
        s_x = self.t2s_proj(s_x)
        return s_x

    def forward(self, s_x: torch.Tensor, t_x: torch.Tensor):
        return self.t2s_cross_attn(s_x, t_x)

    
class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        
        self.clip_ln_1 = norm_layer(dim)
        self.clip_attn = nn.MultiheadAttention(dim, num_heads)
        
        self.ln_s2t = norm_layer(dim) # 이건 cross attn 전용 layer norm으로 변경해야 한다.
        self.s2t_cross = CrossAttentionS2T(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        
        self.ln_t2s = norm_layer(dim)
        self.t2s_cross = CrossAttentionT2S(dim, num_heads)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # Time path
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
        # Space path
        self.clip_ln_2 = norm_layer(dim)
        self.clip_mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(dim, dim * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(dim * 4, dim))
        ]))

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_3 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2, self.gamma_3 = None, None, None
            
    def clip_attention(self, x):
        return self.clip_attn(x, x, x, need_weights=False, attn_mask=None)[0]

    def forward(self,s_x, t_x):
        if self.gamma_1 is None:
            s_x = s_x + self.drop_path((self.clip_attention(self.clip_ln_1(s_x)))) # CLIP space attention
            t_x = t_x + self.drop_path(self.attn(self.norm1(t_x))) # VMAE space-time joint attention
            
            #s_x = rearrange(s_x, '(b t) n d -> b t n d', t=16) # cross attention을 위해 shape을 수정해준다. center frame만 쓰니까 잠시 꺼둔다.
            #cls, patches = torch.split(s_x,[1, 196], dim=1)
            
            s_x = s_x + self.drop_path(self.t2s_cross(self.ln_t2s(s_x), t_x))
            
            # cross attn 순서에 대한 ablation study를 해야 할까....?
            #cls = cls + self.drop_path(self.t2s_cross(cls, t_x).unsqueeze(2)) # Cross attention time to space. 이건 잠시 검증을 위해 꺼둔다.
            t_x = t_x + self.drop_path(self.s2t_cross(s_x, self.ln_s2t(t_x))) # Cross attention space to time
            
            #s_x = torch.cat([cls, patches], dim=1) 
            #s_x = rearrange(s_x, 'b t n d -> (b t) n d', t=16) center frame만 쓰니까 잠시 꺼둔다.
            
            s_x = s_x + self.drop_path(self.clip_mlp(self.clip_ln_2(s_x))) # pass CLIP FFN
            t_x = t_x + self.drop_path(self.mlp(self.norm2(t_x))) # pass VMAE FFN
            
        else: # gamma는 쓸일 없으니까 일단 구현하지말자.
            t_x = t_x + self.drop_path(self.gamma_1 * self.attn(self.norm1(t_x)))
            t_x = t_x + self.drop_path(self.gamma_2 * self.cross(s_x, self.norm2(t_x)))
            t_x = t_x + self.drop_path(self.gamma_3 * self.mlp(self.norm3(t_x)))
        return s_x, t_x
      
      
    
class STCrossTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, 
                 img_size=224, 
                 patch_size=16, 
                 in_chans=3, 
                 num_classes=1000, 
                 embed_dim=768, 
                 depth=12,
                 num_heads=12, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, 
                 init_values=0.,
                 use_learnable_pos_emb=False,
                 init_scale=0.,
                 all_frames=16,
                 tubelet_size=2,
                 use_mean_pooling=True,
                 pretrained_cfg = None):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.tubelet_size = tubelet_size
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, num_frames=all_frames, tubelet_size=self.tubelet_size)
        num_patches = self.patch_embed.num_patches
        
        scale = embed_dim ** -0.5
        self.clip_conv1 = nn.Conv2d(in_channels=3, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.clip_class_embedding = nn.Parameter(scale * torch.randn(embed_dim))
        self.clip_positional_embedding = nn.Parameter(scale * torch.randn((img_size // patch_size) ** 2 + 1, embed_dim))
        self.clip_ln_pre = nn.LayerNorm(embed_dim)

        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        else:
            # sine-cosine positional embeddings is on the way
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        self.pos_drop = nn.Dropout(p=drop_rate)


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        
        self.clip_ln_post = nn.LayerNorm(embed_dim)
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.vmae_fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.head = nn.Linear(embed_dim * 2, num_classes) if num_classes > 0 else nn.Identity()

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        trunc_normal_(self.head.weight, std=.02)
        self.apply(self._init_weights)
        
        self.initialize_parameters(embed_dim, depth)

        self.head.weight.data.mul_(init_scale)
        self.head.bias.data.mul_(init_scale)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def initialize_parameters(self, embed_dim, depth):
        proj_std = (embed_dim ** -0.5) * ((2 * depth) ** -0.5)
        attn_std = embed_dim ** -0.5
        fc_std = (2 * embed_dim) ** -0.5
        for block in self.blocks:
            
            nn.init.normal_(block.s2t_cross.s2t_q.weight, std=attn_std)
            nn.init.normal_(block.s2t_cross.s2t_kv.weight, std=attn_std)
            nn.init.normal_(block.s2t_cross.proj.weight, std=proj_std)
            
            nn.init.normal_(block.t2s_cross.t2s_q.weight, std=attn_std)
            nn.init.normal_(block.t2s_cross.t2s_kv.weight, std=attn_std)
            nn.init.normal_(block.t2s_cross.t2s_proj.weight, std=proj_std)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'patch_embed', 'cls_token', 'clip_conv1'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
    
    def reset_fcnorm(self):
        self.vmae_fc_norm = nn.LayerNorm(self.embed_dim)

    def forward_features(self, s_x, t_x):
        #s_x = rearrange(x, 'b c t h w -> (b t) c h w') 이건 잠시 module검증을 위해 center만 사용하자.
        s_x = self.clip_conv1(s_x) # shape = [*, embeddim, grid, grid]
        s_x = s_x.reshape(s_x.shape[0], s_x.shape[1], -1) # [*, embeddim, grid**2]
        s_x = s_x.permute(0, 2, 1) # shape[batch, patchnum, embeddim]
        s_x = torch.cat([self.clip_class_embedding.to(s_x.dtype) + torch.zeros(s_x.shape[0], 1, s_x.shape[-1], dtype=s_x.dtype, device=s_x.device), s_x], dim=1)
        s_x = s_x + self.clip_positional_embedding.to(s_x.dtype)
        s_x = self.clip_ln_pre(s_x)
        
        t_x = self.patch_embed(t_x)
        B, _, _ = t_x.size()

        if self.pos_embed is not None:
            t_x = t_x + self.pos_embed.expand(B, -1, -1).type_as(t_x).to(t_x.device).clone().detach()
        t_x = self.pos_drop(t_x)

        for blk in self.blocks:
            s_x, t_x = blk(s_x, t_x)
            
        # s_x = rearrange(s_x, '(b t) patch dim -> b t patch dim', t=16)
        # s_x = s_x[:, :, 0, :] #cls token pick
        # s_x = s_x.mean(1) #average pooling all frame
        # s_x = self.clip_ln_post(s_x)
        s_x = self.clip_ln_post(s_x[:, 0, :]) # cls token만 뽑는다.
        t_x = self.vmae_fc_norm(t_x.mean(1)) # VideoMAE 최종적으로 normalize해주네.
        
        x = torch.cat([s_x, t_x], dim=1)
        
        # x = (s_x + t_x) / 2 # CLIP output과 VMAE output을 average해준다.
        
        return x


    def forward(self, s_x, t_x):
        x = self.forward_features(s_x, t_x)
        x = self.head(x)
        return x



@register_model
def bidir_cross_vit_small_patch16_224(pretrained=False, **kwargs):
    model = STCrossTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def bidir_cross_vit_base_patch16_224(pretrained=False, **kwargs):
    model = STCrossTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    #model.default_cfg = _cfg()
    return model


@register_model
def bidir_cross_vit_base_patch16_384(pretrained=False, **kwargs):
    model = STCrossTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def bidir_cross_vit_large_patch16_224(pretrained=False, **kwargs):
    model = STCrossTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def bidir_cross_vit_large_patch16_384(pretrained=False, **kwargs):
    model = STCrossTransformer(
        img_size=384, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def bidir_cross_vit_large_patch16_512(pretrained=False, **kwargs):
    model = STCrossTransformer(
        img_size=512, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model