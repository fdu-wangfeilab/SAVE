import torch
import math
import einops
import torch.nn as nn
import random
import numpy as np
from torch.distributions import Normal
from timm.models.vision_transformer import Mlp
from timm.models.vision_transformer import Attention
from itertools import combinations


def modulate(x, shift, scale):
    # 对数据做 shift 和 scale 的方法
    # scale.unsqueeze(1) 将 scale 转换成列向量
    res = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    return res


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class DiTblock(nn.Module):
    # adaLN -> attn -> mlp
    def __init__(self, feature_dim=2000, mlp_ratio=4.0, num_heads=10, **kwargs) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(feature_dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(feature_dim, num_heads=num_heads, qkv_bias=True, **kwargs)

        self.norm2 = nn.LayerNorm(feature_dim, elementwise_affine=False, eps=1e-6)
        approx_gelu = lambda: nn.GELU(approximate="tanh")

        mlp_hidden_dim = int(feature_dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=feature_dim,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(feature_dim, 6 * feature_dim, bias=True)
        )

    def forward(self, x, c):
        # 将 condition 投影到 6 * hiddensize 之后沿列切成 6 份
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        # attention blk
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        # mlp blk 采用 ViT 中实现的版本
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    将 time emb 成 frequency_embedding_size 维，再投影到 hidden_size
    """

    def __init__(self, hidden_size=512, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

        # 将输入 emb 成 frequency_embedding_size 维
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        # 最后一维加 None 相当于拓展一维
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        # 采用 pos emb 之后过 mlp
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class AttentionBlock(nn.Module):
    def __init__(
        self,
        feature_dim=2000,
        mlp_ratio=4.0,
        num_heads=10,
        norm_type="bn",
        affine=True,
        expand_dim=32,
        **kwargs,
    ) -> None:
        super().__init__()

        if norm_type == "bn":
            self.norm1 = nn.BatchNorm1d(feature_dim, affine=affine, eps=1e-6)
            self.norm2 = nn.BatchNorm1d(feature_dim, affine=affine, eps=1e-6)
        elif norm_type == "ln":
            self.norm1 = nn.LayerNorm(expand_dim, elementwise_affine=affine, eps=1e-6)
            self.norm2 = nn.LayerNorm(expand_dim, elementwise_affine=affine, eps=1e-6)

        self.attn = Attention(expand_dim, num_heads=num_heads, qkv_bias=True, **kwargs)
        approx_gelu = lambda: nn.GELU(approximate="tanh")

        mlp_hidden_dim = int(expand_dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=expand_dim,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )

    def forward(self, x, c=None):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class AttentionEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        depth,
        num_heads,
        enc_dim,
        bn_affine=True,
        norm_type="bn",
        expand_dim=32,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.expand_dim = expand_dim

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
        )

        self.expand_layer = nn.Linear(1, expand_dim)

        self.blks = nn.ModuleList()
        for _ in range(depth):
            self.blks.append(
                AttentionBlock(
                    feature_dim=hidden_dim,
                    expand_dim=expand_dim,
                    mlp_ratio=4.0,
                    num_heads=num_heads,
                    affine=bn_affine,
                    norm_type=norm_type,
                )
            )

        self.var_enc = nn.Sequential(nn.Linear(expand_dim * hidden_dim, enc_dim))

        self.mu_enc = nn.Sequential(nn.Linear(expand_dim * hidden_dim, enc_dim))

    def reparameterize(self, mu, var):
        return Normal(mu, var.sqrt()).rsample()

    def forward(self, x, y=None):
        h = self.input_layer(x).unsqueeze(-1)
        h = self.expand_layer(h)

        for blk in self.blks:
            h = blk(h)

        h = einops.rearrange(h, "c h e -> c (h e)")
        mu = self.mu_enc(h)

        # make sure var>0
        var = torch.clamp(torch.exp(self.var_enc(h)), min=1e-20)
        z = self.reparameterize(mu, var)

        return z, mu, var


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, out_dim):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.linear = nn.Sequential(
            nn.Linear(hidden_size, out_dim, bias=True),
            nn.Sigmoid(),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class AttentionDecoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_heads,
        depth,
        out_dim,
        expand_dim=32,
        is_norm_init=False,
        block_type="adaLN",
        len_col_comb=None,
    ) -> None:
        super().__init__()

        self.block_type = block_type
        self.blks = nn.ModuleList()
        self.input_dim = input_dim
        self.expand_dim = expand_dim

        if len_col_comb:
            col_comb = []
            for i in range(len_col_comb):
                col_comb.extend(list(combinations(range(len_col_comb), i + 1)))
            col_comb = [list(x) if len(x) > 1 else x[0] for x in col_comb]
            print("total col comb is:", len(col_comb))
            self.col_comb = col_comb

        if block_type == "dsbn":
            pass
        elif block_type == "adaLN":
            self.reshape_layer = nn.Linear(1, expand_dim)

            self.batch_embedder = TimestepEmbedder(hidden_size=expand_dim)

            for _ in range(depth):
                self.blks.append(
                    DiTblock(
                        feature_dim=expand_dim,
                        mlp_ratio=4.0,
                        num_heads=num_heads,
                    )
                )

        self.out_layer = nn.Sequential(
            nn.Linear(hidden_dim * expand_dim, out_dim), nn.Sigmoid()
        )

        if is_norm_init:
            self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize batch embedding MLP:
        nn.init.normal_(self.batch_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.batch_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        # nn.init.constant_(self.out_layer.adaLN_modulation[-1].weight, 0)
        # nn.init.constant_(self.out_layer.adaLN_modulation[-1].bias, 0)
        # nn.init.constant_(self.out_layer.linear[0].weight, 0)
        # nn.init.constant_(self.out_layer.linear[0].bias, 0)

    def forward(
        self,
        x,
        y: torch.Tensor = None,
        col_msk_threshold=0.8,
        row_msk_threshold=None, # cancel row msk
        target_msk_col=None,
    ):
        if self.block_type == "adaLN":
            # split cols
            y_split = y.unbind(1)
            # row concat and encoding
            c = self.batch_embedder(torch.hstack(y_split))

            # split to ori shape
            c_split = c.split(x.shape[0], dim=0)
            c_stk = torch.stack(c_split, dim=0)

            if random.random() < col_msk_threshold:
                # random mask condition
                if target_msk_col is None:
                    idx = random.choice(self.col_comb)
                    c_stk[idx] = 0
                else:
                    c_stk[target_msk_col] = 0

            c = torch.sum(c_stk, dim=0)

            if row_msk_threshold is not None:
                row_msk = torch.rand(c.shape[0], 1).to(c.device) > row_msk_threshold
                row_msk = row_msk.expand(c.shape)
                c = c * row_msk

            h = self.reshape_layer(x.unsqueeze(-1))
            for blk in self.blks:
                h = blk(h, c)

        elif self.block_type == "dsbn":
            h = self.input_layer(x, y)
            h = self.proj(h)
            for blk in self.blks:
                h = blk(h)

        h = einops.rearrange(h, "c h e -> c (h e)")
        return self.out_layer(h)


class VAE(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        enc_num_heads,
        dec_num_heads,
        enc_depth,
        dec_depth,
        enc_dim=10,
        expand_dim=32,
        enc_affine=True,
        enc_norm_type="bn",
        dec_norm_init=False,
        dec_blk_type="adaLN",
        is_initialize=False,
        len_col_comb=None,
    ) -> None:
        super().__init__()
        # for parameter record
        self.hidden_dim = hidden_dim
        self.latent_dim = enc_dim

        self.enc_heads = enc_num_heads
        self.dec_heads = dec_num_heads

        self.enc_depth = enc_depth
        self.dec_depth = dec_depth

        self.enc_affine = enc_affine
        self.enc_norm_type = enc_norm_type

        self.dec_blk_type = dec_blk_type

        self.encoder = AttentionEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            expand_dim=expand_dim,
            num_heads=enc_num_heads,
            depth=enc_depth,
            enc_dim=enc_dim,
            bn_affine=enc_affine,
            norm_type=enc_norm_type,
        )

        self.decoder = AttentionDecoder(
            input_dim=enc_dim,
            hidden_dim=enc_dim,
            expand_dim=expand_dim,
            num_heads=dec_num_heads,
            depth=dec_depth,
            out_dim=input_dim,
            is_norm_init=dec_norm_init,
            block_type=dec_blk_type,
            len_col_comb=len_col_comb,
        )
        if is_initialize:
            self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
