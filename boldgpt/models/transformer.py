from typing import Callable, List, Literal, Optional, Tuple, Union

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
        kv_cache: Optional[torch.Tensor] = None,
        return_kv: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, N, C = x.shape
        if context is None:
            context = x
        M = context.size(1)

        # (B, num_heads, N, head_dim)
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # (2, B, num_heads, M, head_dim)
        kv = (
            self.kv(context)
            .reshape(B, M, 2, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        if kv_cache is not None:
            kv = torch.cat([kv_cache, kv], dim=3)
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

        if return_kv:
            return x, kv
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

        # Decoding related cache and flags
        self._kv_cache: Optional[torch.Tensor] = None
        self._is_decoding = False
        self._use_cache = True

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        assert (
            not self._is_decoding or is_causal
        ), "Can only decode with causal attention"

        if self._is_decoding and self._use_cache:
            assert (
                self._kv_cache is None or x.shape[1] == 1
            ), "Can only decode one token at a time with caching"

            y, kv = self.attn(
                self.norm1(x),
                attn_mask=attn_mask,
                is_causal=is_causal and self._kv_cache is None,
                kv_cache=self._kv_cache,
                return_kv=True,
            )
            self._kv_cache = kv.detach()
            y = self.ls1(y)
        else:
            y = self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)
            y = self.ls1(y)
        x = x + self.drop_path1(y)

        if context is not None:
            y = self.ls2(self.cross(self.norm2(x), context=context))
            x = x + self.drop_path2(y)

        y = self.ls3(self.mlp(self.norm3(x)))
        x = x + self.drop_path3(y)
        return x

    def decoding(self, mode: bool = True, use_cache: bool = True):
        if mode:
            self._use_cache = use_cache
        else:
            self._kv_cache = None
        self._is_decoding = mode


class TokenDropout(nn.Dropout1d):
    """
    Dropout tokens without scaling by `1 / (1 - p)`.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = super().forward(input)
        if self.training:
            output = (1 - self.p) * output
        return output


class Transformer(nn.Module):
    def __init__(
        self,
        num_patches: int,
        in_features: int,
        num_subs: int = 1024,
        num_registers: int = 0,
        num_classes: int = 4096,
        global_pool: Optional[Literal["avg", "token", "reg"]] = None,
        embed_dim: int = 768,
        context_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        with_sub_embed: bool = True,
        with_next_pos: bool = True,
        with_cross: bool = False,
        is_causal: bool = True,
        is_masked: bool = False,
        drop_rate: float = 0.0,
        sub_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        assert (
            global_pool != "reg" or num_registers > 0
        ), "Must set num_registers > 0 to use 'reg' global pooling"

        self.num_patches = num_patches
        self.in_features = in_features
        self.num_subs = num_subs
        self.num_registers = num_registers
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.embed_dim = embed_dim
        self.context_dim = context_dim
        self.with_sub_embed = with_sub_embed
        self.with_next_pos = with_next_pos
        self.with_cross = with_cross
        self.is_causal = is_causal
        self.is_masked = is_masked

        self.patch_embed = nn.Linear(in_features, embed_dim)
        if self.with_cross:
            self.cross_embed = nn.Linear(context_dim, embed_dim)
        else:
            self.register_module("cross_embed", None)

        self.group_token = nn.Parameter(torch.empty(1, 1, embed_dim))
        if with_sub_embed:
            self.sub_embed = nn.Parameter(torch.empty(num_subs, 1, embed_dim))
        else:
            self.register_parameter("sub_embed", None)
        self.pos_embed = nn.Parameter(torch.empty(num_patches, embed_dim))
        if with_next_pos:
            self.next_pos_query = nn.Parameter(torch.empty(num_patches, embed_dim))
            self.eos_query = nn.Parameter(torch.empty(1, embed_dim))
        else:
            self.register_parameter("next_pos_query", None)
            self.register_parameter("eos_query", None)
        if num_registers > 0:
            self.reg_embed = nn.Parameter(torch.empty(num_registers, embed_dim))
        else:
            self.register_parameter("reg_embed", None)
        self.sub_drop = TokenDropout(p=sub_drop_rate)
        if is_masked:
            self.mask_token = nn.Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.register_parameter("mask_token", None)

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

        use_fc_norm = self.global_pool in {"avg", "reg"}
        self.norm = nn.Identity() if use_fc_norm else nn.LayerNorm(embed_dim)
        self.fc_norm = nn.LayerNorm(embed_dim) if use_fc_norm else nn.Identity()

        # Classifier Head
        self.head_drop = nn.Dropout(drop_rate)
        if num_classes > 0:
            self.head = nn.Linear(self.embed_dim, num_classes)
        else:
            self.head = nn.Identity()

        self.init_weights()

        self._is_decoding = False
        self._was_training = False

    def init_weights(self):
        if self.is_masked:
            trunc_normal_(self.mask_token, std=0.02)
        nn.init.zeros_(self.group_token)
        if self.with_sub_embed:
            trunc_normal_(self.sub_embed, std=0.02)
        trunc_normal_(self.pos_embed, std=0.02)
        if self.with_next_pos:
            trunc_normal_(self.next_pos_query, std=0.02)
            trunc_normal_(self.eos_query, std=0.02)
        if self.num_registers > 0:
            trunc_normal_(self.reg_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _mask_pos(self, x: torch.Tensor, bool_masked_pos: torch.Tensor) -> torch.Tensor:
        # token masking following BEiT
        assert self.is_masked, "model must have is_masked=True"
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
        offset: Optional[int] = None,
    ) -> torch.Tensor:
        assert (
            order is None or not self.is_causal or self.with_next_pos
        ), "Must set with_next_pos=True for non-default patch order"
        assert (
            sub_indices is None or self.with_sub_embed
        ), "Must set with_sub_embed=True to use sub_indices"
        # position of first token, -1 means start with subject token
        assert offset is None or -1 <= offset < self.num_patches, "Invalid offset"
        B = x.size(0)

        # learned position encoding
        pos_embed = self.pos_embed
        if order is not None:
            pos_embed = pos_embed[order]
        pos_embed = pos_embed.expand(B, -1, -1)
        # slice position embedding the start of the x subsequence
        # only relevant during cached decoding
        if offset is not None:
            start = max(offset, 0)
            pos_embed = pos_embed[:, start : start + x.size(1)]
        x = x + pos_embed

        if sub_indices is not None:
            # subject encoding
            sub_token = self.sub_drop(self.sub_embed[sub_indices])
            sub_token = self.group_token + sub_token
        else:
            # group token only
            sub_token = self.group_token.expand(B, -1, -1)
        if offset is None or offset < 0:
            x = torch.cat([sub_token, x], dim=1)

        # learned next position query (for shuffled orders)
        if self.with_next_pos:
            next_pos_query = self.next_pos_query
            if order is not None:
                next_pos_query = next_pos_query[order]
            next_pos_query = next_pos_query.expand(B, -1, -1)
            eos_query = self.eos_query.expand(B, -1, -1)
            next_pos_query = torch.cat([next_pos_query, eos_query], dim=1)
            if offset is not None:
                start = offset + 1
                next_pos_query = next_pos_query[:, start : start + x.size(1)]
            x = x + next_pos_query

        # append registers
        if self.num_registers > 0:
            reg_embed = self.reg_embed.expand(B, -1, -1)
            x = torch.cat([x, reg_embed], dim=1)
        return x

    def _get_attn_mask(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        L = x.size(-2)
        device = x.device

        # Allow global attention between sequence and registers
        if self.is_causal and self.num_registers > 0:
            attn_mask = torch.ones(L, L, dtype=torch.bool, device=device).tril_()
            attn_mask[:, -self.num_registers :] = True
            attn_mask[-self.num_registers :, :] = True
        else:
            attn_mask = None
        return attn_mask

    def forward_features(
        self,
        x: torch.Tensor,
        sub_indices: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        order: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.Tensor] = None,
        offset: Optional[int] = None,
    ) -> torch.Tensor:
        assert (
            context is None or self.with_cross
        ), "Must set with_cross=True to use context"
        x = self.patch_embed(x)
        if context is not None:
            context = self.cross_embed(context)

        if bool_masked_pos is not None:
            x = self._mask_pos(x, bool_masked_pos)
        x = self._pos_embed(x, sub_indices, order, offset=offset)

        attn_mask = self._get_attn_mask(x)
        is_causal = self.is_causal and attn_mask is None

        for block in self.blocks:
            x = block(x, context=context, is_causal=is_causal, attn_mask=attn_mask)

        x = self.norm(x)
        return x

    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        if self.global_pool == "avg":
            x = x[:, 1:].mean(dim=1)
        elif self.global_pool == "reg":
            x = x[:, -self.num_registers :].mean(dim=1)
        elif self.global_pool:
            x = x[:, 0]
        x = self.fc_norm(x)
        x = self.head_drop(x)
        x = self.head(x)
        return x

    def forward(
        self,
        patches: torch.Tensor,
        sub_indices: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        order: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.Tensor] = None,
        offset: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            patches: input patches (B, N, D)
            sub_indices: subject indices (B,)
            context: context features (B, M, C)
            order: token order (B, N) or (N,)
            bool_masked_pos: masked token positions (B, N)
            offset: position of first token, -1 means start with subject token

        Returns:
            output tensor (B, N+1, C), where the +1 is due to the prepended subject
            token.
        """
        x = self.forward_features(
            patches,
            sub_indices=sub_indices,
            context=context,
            order=order,
            bool_masked_pos=bool_masked_pos,
            offset=offset,
        )

        x = self.forward_head(x)
        return x

    def no_decay_keys(self) -> List[str]:
        """
        Return a list of parameter names that should not be weight decayed.
        """
        # Don't decay biases, layernorms, or position embeddings
        # Combination of what's done in timm and nanoGPT
        keys = [
            name
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
                "reg_embed",
            }
        ]
        return keys

    def decoding(self, mode: bool = True, use_cache: bool = True):
        if mode:
            self._was_training = self.training
            self.eval()
        else:
            self.train(self._was_training)
        self._is_decoding = mode

        for block in self.blocks:
            block.decoding(mode=mode, use_cache=use_cache)

    def extra_repr(self) -> str:
        return (
            f"num_registers={self.num_registers}, "
            f"with_sub_embed={self.with_sub_embed}, "
            f"with_next_pos={self.with_next_pos}, "
            f"with_cross={self.with_cross}, "
            f"is_caual={self.is_causal}, "
            f"is_masked={self.is_masked}"
        )
