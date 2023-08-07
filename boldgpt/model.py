from typing import Callable, Dict, Optional

import torch
import torch.nn.functional as F
from timm.layers import DropPath, Mlp, trunc_normal_
from torch import nn

Layer = Callable[..., nn.Module]


class Attention(nn.Module):
    """
    An attention layer with support for cross and causal attention.

    Based on timm vision_transformer.
    """

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
        is_causal: bool = False,
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

        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop.p,
            is_causal=is_causal,
            attn_mask=attn_mask,
        )

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
        is_causal: bool = False,
    ) -> torch.Tensor:
        y = self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal))
        x = x + self.drop_path1(y)

        if context is not None:
            y = self.ls2(self.cross(self.norm2(x), context))
            x = x + self.drop_path2(y)

        y = self.ls3(self.mlp(self.norm3(x)))
        x = x + self.drop_path3(y)
        return x


class TokenDropout(nn.Dropout1d):
    """
    Dropout tokens without scaling by `1 / (1 - p)`.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = super().forward(input)
        if self.training:
            output = (1 - self.p) * output
        return output


class BoldGPT(nn.Module):
    def __init__(
        self,
        num_patches: int,
        in_features: int,
        num_subs: int = 8,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        with_cross: bool = False,
        is_decoder: bool = False,
        drop_rate: float = 0.0,
        sub_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.num_patches = num_patches
        self.in_features = in_features
        self.num_subs = num_subs
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.is_decoder = is_decoder

        self.patch_embed = nn.Linear(in_features, embed_dim)

        self.mask_token = nn.Parameter(torch.empty(1, 1, embed_dim))
        self.group_token = nn.Parameter(torch.empty(1, 1, embed_dim))
        self.sub_embed = nn.Parameter(torch.empty(num_subs, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.empty(num_patches, embed_dim))
        self.next_pos_query = nn.Parameter(torch.empty(num_patches, embed_dim))
        self.eos_query = nn.Parameter(torch.empty(1, embed_dim))
        self.sub_drop = TokenDropout(p=sub_drop_rate)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    proj_drop=proj_drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    cross_attn=with_cross,
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        # Classifier Head
        self.head_drop = nn.Dropout(drop_rate)
        if num_classes > 0:
            self.head = nn.Linear(self.embed_dim, num_classes)
        else:
            self.head = nn.Identity()

        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.group_token, std=1e-6)
        trunc_normal_(self.sub_embed, std=0.02)
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.next_pos_query, std=0.02)
        trunc_normal_(self.eos_query, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _mask_pos(
        self,
        x: torch.Tensor,
        bool_masked_pos: Optional[torch.Tensor] = None,
    ):
        if bool_masked_pos is None:
            return x

        # token masking following BEiT
        B, N, _ = x.shape
        mask_token = self.mask_token.expand(B, N, -1)

        # replace the masked visual tokens by mask_token
        w = bool_masked_pos.unsqueeze(-1).type_as(mask_token)
        x = x * (1 - w) + mask_token * w
        return x

    def _pos_embed(
        self,
        x: torch.Tensor,
        sub_indices: Optional[torch.Tensor] = None,
        order: Optional[torch.Tensor] = None,
    ):
        # learned position encoding
        pos_embed = self.pos_embed
        if order is not None:
            pos_embed = pos_embed[order]
        x = x + pos_embed

        if sub_indices is not None:
            # subject encoding
            sub_token = self.sub_drop(self.sub_embed[sub_indices])
            sub_token = self.group_token + sub_token
        else:
            # group token only
            sub_token = self.group_token.expand(x.size(0), -1, -1)
        x = torch.cat([sub_token, x], dim=1)

        # learned next position query (for shuffled orders)
        next_pos_query = self.next_pos_query
        if order is not None:
            next_pos_query = next_pos_query[order]
        next_pos_query = torch.cat([next_pos_query, self.eos_query])
        x = x + next_pos_query
        return x

    def forward_features(
        self,
        x: torch.Tensor,
        sub_indices: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        order: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.patch_embed(x)

        x = self._mask_pos(x, bool_masked_pos)
        x = self._pos_embed(x, sub_indices, order)

        for block in self.blocks:
            x = block(x, context=context, is_causal=self.is_decoder)

        x = self.norm(x)
        return x

    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        x = self.head_drop(x)
        x = self.head(x)
        return x

    def forward(
        self,
        x: torch.Tensor,
        sub_indices: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        order: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: input tokens (B, N, P)
            sub_indices: subject indices (B,)
            context: context features (B, M, C)
            order: token order (B, N) or (N,)
            bool_masked_pos: masked token positions (B, N)
        """
        x = self.forward_features(
            x,
            sub_indices=sub_indices,
            context=context,
            order=order,
            bool_masked_pos=bool_masked_pos,
        )
        x = self.forward_head(x)
        return x

    def no_decay_named_parameters(self) -> Dict[str, nn.Parameter]:
        """
        Return a dict of named parameters that should not be weight decayed.
        """
        # Don't decay biases, layernorms, or position embeddings
        # Combination of what's done in timm and nanoGPT
        params = {
            name: p
            for name, p in self.named_parameters()
            if p.ndim < 2
            or name
            in {
                "mask_token",
                "group_token",
                "sub_embed",
                "pos_embed",
                "next_pos_query",
                "eos_query",
            }
        }
        return params
