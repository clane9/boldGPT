from typing import Callable, Optional

import torch
import torch.nn.functional as F
from timm.layers import DropPath, Mlp, trunc_normal_, use_fused_attn
from torch import nn
from torch.jit import Final

from .order import Order
from .patching import MaskedPatchify

Layer = Callable[..., nn.Module]


class Attention(nn.Module):
    """
    An attention layer with support for cross and causal attention.

    Based on timm vision_transformer.
    """

    fused_attn: Final[bool]

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: Layer = nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = use_fused_attn()

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, C = x.shape
        if context is None:
            context = x
        M = context.size(1)

        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = (
            self.kv(context)
            .reshape(B, M, 2, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p,
                attn_mask=attn_mask,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if attn_mask is not None:
                attn = attn.masked_fill(~attn_mask, float("-inf"))
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: Layer = nn.GELU,
        norm_layer: Layer = nn.LayerNorm,
        mlp_layer: Layer = Mlp,
        cross_attn: bool = False,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        if cross_attn:
            self.norm2 = norm_layer(dim)
            self.cross = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                norm_layer=norm_layer,
            )
            self.ls2 = (
                LayerScale(dim, init_values=init_values)
                if init_values
                else nn.Identity()
            )
            self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm3 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls3 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path3 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        y = self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask))
        x = x + self.drop_path1(y)

        if context is not None:
            y = self.ls2(self.cross(self.norm2(x), context, attn_mask=attn_mask))
            x = x + self.drop_path2(y)

        y = self.ls3(self.mlp(self.norm3(x), attn_mask=attn_mask))
        x = x + self.drop_path3(y)
        return x


class BoldGPT(nn.Module):
    def __init__(
        self,
        mask: torch.Tensor,
        patch_size: int = 16,
        num_subs: int = 8,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        is_decoder: bool = False,
        drop_rate: float = 0.0,
        sub_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.num_subs = num_subs
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.is_decoder = is_decoder
        self.num_prefix_tokens = 1
        self.grad_checkpointing = False

        self.patchify = MaskedPatchify(mask, patch_size, in_chans=1)
        self.patch_embed = nn.Linear(self.patchify.dim, embed_dim)
        self.num_patches = num_patches = self.patchify.num_patches

        self.group_token = nn.Parameter(torch.empty(1, 1, embed_dim))
        self.sub_embed = nn.Parameter(torch.empty(num_subs, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.empty(num_patches, embed_dim))
        self.next_pos_embed = nn.Parameter(torch.empty(num_patches + 1, embed_dim))
        self.sub_drop = nn.Dropout(p=sub_drop_rate)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    proj_drop=proj_drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    cross_attn=is_decoder,
                )
                for i in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)

        # Classifier Head
        self.head_drop = nn.Dropout(drop_rate)
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.group_token, std=1e-6)
        trunc_normal_(self.sub_embed, std=0.02)
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.next_pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _pos_embed(
        self,
        x: torch.Tensor,
        order: Order,
        sub_indices: Optional[torch.Tensor] = None,
    ):
        # learned position encoding
        x = x + self.pos_embed
        # learned next position queries
        x = x + self.next_pos_embed[order.next_indices]

        if sub_indices is not None:
            # subject encoding
            sub_token = self.sub_embed[sub_indices]
            sub_token = self.group_token + self.sub_drop(sub_token)
        else:
            # group token only
            sub_token = self.group_token.expand(x.size(0), -1, -1)

        # add next position query to subject token
        sub_token = sub_token + self.next_pos_embed[order.order[:, :1]]

        x = torch.cat((sub_token, x), dim=1)
        return x

    def forward_features(
        self,
        x: torch.Tensor,
        sub_indices: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        order: Optional[Order] = None,
    ) -> torch.Tensor:
        B = x.size(0)
        device = x.device

        if order is None:
            order = Order.shuffled(B, self.num_patches, device=device)

        x = self.patchify(x)
        x = self.patch_embed(x)
        x = self._pos_embed(x, order, sub_indices=sub_indices)

        for block in self.blocks:
            x = block(x, context=context, attn_mask=order.attn_mask)

        x = self.norm(x)
        return x

    def forward_head(self, x: torch.Tensor):
        x = x[:, : self.num_patches]
        x = self.head_drop(x)
        x = self.head(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x
