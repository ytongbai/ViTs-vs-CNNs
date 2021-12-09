# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from functools import partial

from vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_

from attacker import PGDAttacker, NoOpAttacker



__all__ = [
    'deit_tiny_patch16_224', 'deit_small_patch16_224', 'deit_base_patch16_224',
    'deit_tiny_distilled_patch16_224', 'deit_small_distilled_patch16_224',
    'deit_base_distilled_patch16_224', 'deit_base_patch16_384',
    'deit_base_distilled_patch16_384', 
    'deit_small_patch16_224_adv', 
]


def to_status(m, status):
    if hasattr(m, 'batch_type'):
        m.batch_type = status


to_clean_status = partial(to_status, status='clean')
to_adv_status = partial(to_status, status='adv')
to_mix_status = partial(to_status, status='mix')


class DistilledVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.head_dist.apply(self._init_weights)

    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0], x[:, 1]

    def forward(self, x):
        x, x_dist = self.forward_features(x)
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        if self.training:
            return x, x_dist
        else:
            # during inference, return the average of both classifier predictions
            return (x + x_dist) / 2

        
class AdvVisionTransformer(VisionTransformer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None,  attacker=NoOpAttacker()):
        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes, embed_dim=embed_dim, depth=depth,
    num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                 drop_path_rate=drop_path_rate, hybrid_backbone=hybrid_backbone)
        norm_layer=nn.LayerNorm
        self.attacker = attacker 
        self.mix = False
        self.sing = False
        self.mixup_fn = False
#         self.iter = 0
    def set_mixup_fn(self, mixup):
        self.mixup_fn = mixup
    def set_attacker(self, attacker):
        self.attacker = attacker
    
#     def set_iters(self, iter_):
#         self.iter = iter_ 
    
    def set_mix(self, mix):
        self.mix = mix
    def set_sing(self, sing):
        self.sing = sing
    def forward(self, x, labels=None):
        # advprop
        if not self.sing: 
            training = self.training
            input_len = len(x)
            if training:
                self.eval()
                self.apply(to_adv_status)
                if isinstance(self.attacker, NoOpAttacker):
                    images = x
                    targets = labels 
                else:
                    aux_images, _ = self.attacker.attack(x, labels, self._forward_impl, False)
                    images = torch.cat([x, aux_images], dim=0)
                    targets = torch.cat([labels, labels], dim=0)
                self.train()
                if self.mix:
                    self.apply(to_mix_status)
                    # check shape
#                     print(labels.shape, images.shape, targets.shape, aux_images.shape)
                    if len(labels.shape) == 2:
                        return self._forward_impl(images).view(2, input_len, -1).transpose(1, 0), targets.view(2, input_len, -1).transpose(1, 0)
                    else:
                        return self._forward_impl(images).view(2, input_len, -1).transpose(1, 0), targets.view(2, input_len).transpose(1, 0)
                else:
                    self.apply(to_clean_status)
                    return self._forward_impl(images), targets
            else:
                images = x
                targets = labels
                return self._forward_impl(images), targets
        # advtraining
        else:
            training = self.training
            input_len = len(x)
            if training:
                self.eval()
                self.apply(to_adv_status)
                if isinstance(self.attacker, NoOpAttacker):
                    images = x
                    targets = labels 
                else:
                    aux_images, _ = self.attacker.attack(x, labels, self._forward_impl, True, True, self.mixup_fn)
                    images =  aux_images
                    targets = labels
                self.train()
                self.apply(to_clean_status)
                return self._forward_impl(images), targets
            else:
                if isinstance(self.attacker, NoOpAttacker):
                    images = x
                    targets = labels 
                else:
                    aux_images, _ = self.attacker.attack(x, labels, self._forward_impl, True, False, False)
                    images =  aux_images
                    targets = labels
                if targets is None:
                    return self._forward_impl(images)
                return self._forward_impl(images), targets

        
# [64, 197, 768]
class BGN(nn.Module):
    def __init__(self, num_channels, num_groups=2,  eps = 1e-05, affine= True):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        self.groupnorm = nn.GroupNorm(num_channels=self.num_channels, num_groups=self.num_groups, eps=self.eps)
    
    def forward(self, x):
        x = x.permute(1, 2, 0)
        x = self.groupnorm(x)
        x = x.permute(2, 0, 1)
        return x
        

class GN(nn.Module):
    def __init__(self, num_groups, num_channels, eps = 1e-05, affine= True):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        self.groupnorm = nn.GroupNorm(num_channels=self.num_channels, num_groups=self.num_groups, eps=self.eps)
    
    def forward(self, x):
        n = x.shape[0]
        p = x.shape[1]
        c = x.shape[2]
        x = x.reshape(-1, c)
        x = self.groupnorm(x)
        x = x.reshape(n, p, c)

        return x  
        
        
class BN(nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.batchnorm = nn.BatchNorm1d(num_features=self.num_features, eps=self.eps, momentum=self.momentum, affine=self.affine, track_running_stats=self.track_running_stats)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.batchnorm(x)
        x = x.permute(0, 2, 1)
        return x
        
    
@register_model
def deit_tiny_patch16_224(pretrained=False, **kwargs):
    norm = kwargs['norm']
    del kwargs['norm']
    if norm == 'layer':
        model = VisionTransformer(
            patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    elif norm == 'group':
        model = VisionTransformer(
            patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
            norm_layer = lambda planes: GN(num_groups=2, num_channels=planes, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_tiny_patch16_224_adv(pretrained=False, **kwargs):
    model = AdvVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True, **kwargs)
    model.default_cfg = _cfg()
    return model



@register_model
def deit_small_patch16_224_adv(pretrained=False, **kwargs):
    model = AdvVisionTransformer(
            patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True, **kwargs)
    model.default_cfg = _cfg()
    '''
    norm = kwargs['norm']
    del kwargs['norm']
    if norm == 'layer':
        model = AdvVisionTransformer(
            patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    elif norm == 'bgn':
        model = AdvVisionTransformer(
            patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
            norm_layer = lambda planes: BGN(num_groups=2, num_channels=planes, eps=1e-6), **kwargs)
    elif norm == 'group':
        model = AdvVisionTransformer(
            patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
            norm_layer = lambda planes: GN_2(num_groups=2, num_channels=planes, eps=1e-6), **kwargs)
    elif norm == 'group_4':
        model = AdvVisionTransformer(
            patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
            norm_layer = lambda planes: GN_2(num_groups=4, num_channels=planes, eps=1e-6), **kwargs)
    elif norm == 'batch':
        model = AdvVisionTransformer(
            patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
            norm_layer = lambda planes: BN(num_features=planes, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), **kwargs)
    
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    '''
    return model



@register_model
def deit_small_patch16_224_ori(pretrained=False, **kwargs):
    norm = kwargs['norm']
    del kwargs['norm']
    if norm == 'layer':
        model = VisionTransformer(
            patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    elif norm == 'bgn':
        model = VisionTransformer(
            patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
            norm_layer = lambda planes: BGN(num_groups=2, num_channels=planes, eps=1e-6), **kwargs)
    elif norm == 'group':
        model = VisionTransformer(
            patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
            norm_layer = lambda planes: GN_2(num_groups=2, num_channels=planes, eps=1e-6), **kwargs)
    elif norm == 'group_4':
        model = VisionTransformer(
            patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
            norm_layer = lambda planes: GN_2(num_groups=4, num_channels=planes, eps=1e-6), **kwargs)
    elif norm == 'batch':
        model = VisionTransformer(
            patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
            norm_layer = lambda planes: BN(num_features=planes, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), **kwargs)
    
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_patch16_224(pretrained=False, **kwargs):
    norm = kwargs['norm']
    del kwargs['norm']
    if norm == 'layer':
        model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    elif norm == 'bgn':
        model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer = lambda planes: BGN(num_groups=2, num_channels=planes, eps=1e-6), **kwargs)
    elif norm == 'group':
        model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer = lambda planes: GN(num_groups=2, num_channels=planes, eps=1e-6), **kwargs)
    elif norm == 'batch':
        model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer = lambda planes: BN(num_features=planes, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_tiny_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_small_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_distilled_patch16_384(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model
