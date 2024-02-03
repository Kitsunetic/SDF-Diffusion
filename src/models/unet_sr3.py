import math
from copy import deepcopy
from inspect import isfunction

import torch as th
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from src.utils.indexing import unsqueeze_as


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def dropout_nd(dims, use_nd, p=0.0, *args, **kwargs):
    if p == 0.0:
        return nn.Identity()
    elif not use_nd:
        return nn.Dropout(*args, **kwargs)
    elif dims == 1:
        return nn.Dropout(*args, **kwargs)
    elif dims == 2:
        return nn.Dropout2d(*args, **kwargs)
    elif dims == 3:
        return nn.Dropout3d(*args, **kwargs)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def Normalization(group, ch):
    if group == -1:
        group = ch
    # return nn.GroupNorm(group, ch)
    return GroupNorm32(group, ch)
    # return {1: nn.InstanceNorm1d, 2: nn.InstanceNorm2d, 3: nn.InstanceNorm3d}[dims](ch)
    # return {1: nn.BatchNorm1d, 2: nn.BatchNorm2d, 3: nn.BatchNorm3d}[dims](ch)
    # return nn.BatchNorm2d(ch)


class DiscreteTimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = th.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = th.exp(-emb)
        pos = th.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = th.stack([th.sin(emb), th.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t):
        emb = self.timembedding(t)
        return emb


class ClassEmbedding(nn.Module):
    def __init__(self, num_classes, d_model, dim):
        super().__init__()
        self.emb = nn.Embedding(num_classes, d_model)
        self.fc = nn.Sequential(
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )

    def forward(self, c):
        emb = self.emb(c)
        emb = self.fc(emb)
        return emb


# PositionalEncoding Source： https://github.com/lmnt-com/wavegrad/blob/master/src/wavegrad/model.py
class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2
        step = th.arange(count, dtype=noise_level.dtype, device=noise_level.device) / count
        encoding = noise_level.unsqueeze(1) * th.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = th.cat([th.sin(encoding), th.cos(encoding)], dim=-1)
        return encoding


class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(nn.Linear(in_channels, out_channels * (1 + self.use_affine_level)))

    def forward(self, x, noise_embed):
        if self.use_affine_level:
            gamma, beta = unsqueeze_as(self.noise_func(noise_embed), x).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + unsqueeze_as(self.noise_func(noise_embed), x)
        return x


class Swish(nn.Module):
    def forward(self, x):
        return x * th.sigmoid(x)


class Upsample(nn.Module):
    def __init__(self, dims, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = conv_nd(dims, dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dims, dim):
        super().__init__()
        self.conv = conv_nd(dims, dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


# building block modules


class Block(nn.Module):
    def __init__(self, dims, dim, dim_out, groups=32, dropout=0, use_nd_dropout=False):
        super().__init__()
        self.block = nn.Sequential(
            Normalization(groups, dim),
            Swish(),
            dropout_nd(dims, use_nd_dropout, dropout),
            conv_nd(dims, dim, dim_out, 3, padding=1),
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(
        self,
        dims,
        dim,
        dim_out,
        noise_level_emb_dim=None,
        dropout=0,
        use_affine_level=False,
        norm_groups=32,
        use_nd_dropout=False,
    ):
        super().__init__()
        self.noise_func = None
        if noise_level_emb_dim is not None:
            self.noise_func = FeatureWiseAffine(noise_level_emb_dim, dim_out, use_affine_level)

        self.block1 = Block(dims, dim, dim_out, groups=norm_groups, use_nd_dropout=use_nd_dropout)
        self.block2 = Block(dims, dim_out, dim_out, groups=norm_groups, dropout=dropout, use_nd_dropout=use_nd_dropout)
        self.res_conv = conv_nd(dims, dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.block1(x)
        if self.noise_func is not None:
            h = self.noise_func(h, time_emb)
        h = self.block2(h)
        return h + self.res_conv(x)


class SelfAttention(nn.Module):
    def __init__(self, dims, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        self.norm = Normalization(norm_groups, in_channel)
        self.qkv = conv_nd(dims, in_channel, in_channel * 3, 1, bias=False)
        self.out = conv_nd(dims, in_channel, in_channel, 1)

    def forward(self, input):
        d = input.size(1)
        n_head = self.n_head
        head_dim = d // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm)  # b (3 d) ...
        qkv = rearrange(qkv, "b (h d) ... -> b h d (...)", h=n_head).contiguous()
        query, key, value = qkv.chunk(3, dim=2)  # each (b h d (...))

        attn = th.einsum("b h d m, b h d n -> b h m n", query, key) / math.sqrt(d)
        attn = th.softmax(attn, dim=-1)
        out = th.einsum("b h m n, b h d m -> b h d n", attn, value)
        out = self.out(out.view(norm.shape).contiguous())
        # attn = th.einsum("bnchw, bncyx -> bnhwyx", query, key).contiguous() / math.sqrt(d)
        # attn = attn.view(b, n_head, height, width, -1)
        # attn = th.softmax(attn, -1)
        # attn = attn.view(b, n_head, height, width, height, width)
        # out = th.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        # out = self.out(out.view(b, d, height, width))

        return out + input


class CMlp(nn.Module):
    def __init__(self, dims, in_ch, ch=None, out_ch=None, drop=0.0, use_nd_dropout=False):
        super().__init__()
        out_ch = out_ch or in_ch
        ch = ch or in_ch
        self.fc1 = conv_nd(dims, in_ch, ch, 1)
        self.act = nn.GELU()
        self.fc2 = conv_nd(dims, ch, out_ch, 1)
        self.drop = dropout_nd(dims, use_nd_dropout, drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CAttnBlock(nn.Module):
    def __init__(self, dims, in_ch, dropout=0.0, norm_groups=32, use_nd_dropout=False):
        super().__init__()

        self.pos_embed = conv_nd(dims, in_ch, in_ch, 3, padding=1, groups=in_ch)
        self.norm1 = Normalization(norm_groups, in_ch)
        self.norm2 = Normalization(norm_groups, in_ch)
        self.conv1 = conv_nd(dims, in_ch, in_ch, 1)
        self.conv2 = conv_nd(dims, in_ch, in_ch, 1)
        self.attn = conv_nd(dims, in_ch, in_ch, 5, padding=2, groups=in_ch)
        self.proj = CMlp(dims, in_ch, drop=dropout, use_nd_dropout=use_nd_dropout)

    def forward(self, x):
        x = x + self.pos_embed(x)
        x = x + self.conv2(self.attn(self.conv1(self.norm1(x))))
        x = x + self.proj(self.norm2(x))
        return x


class ResnetBlocWithAttn(nn.Module):
    def __init__(
        self,
        dims,
        dim,
        dim_out,
        *,
        noise_level_emb_dim=None,
        norm_groups=32,
        dropout=0,
        with_attn=False,
        with_cattn=False,
        use_affine_level=False,
        use_nd_dropout=False,
    ):
        super().__init__()
        self.with_attn = with_attn
        self.with_cattn = with_cattn
        self.res_block = ResnetBlock(
            dims,
            dim,
            dim_out,
            noise_level_emb_dim,
            norm_groups=norm_groups,
            dropout=dropout,
            use_affine_level=use_affine_level,
            use_nd_dropout=use_nd_dropout,
        )
        if with_attn:
            self.attn = SelfAttention(dims, dim_out, norm_groups=norm_groups)
        if with_cattn:
            self.cattn = CAttnBlock(dims, dim_out, dropout=dropout, norm_groups=norm_groups, use_nd_dropout=use_nd_dropout)

    def forward(self, x, time_emb):
        x = self.res_block(x, time_emb)
        if self.with_attn:
            x = self.attn(x)
        if self.with_cattn:
            x = self.cattn(x)
        return x


class UNet(nn.Module):
    def __init__(
        self,
        dims=2,
        in_channel=6,
        out_channel=3,
        inner_channel=32,
        norm_groups=32,
        channel_mults=[1, 2, 4, 8, 8],
        attn_res=[8],
        cattn_res=[],
        res_blocks=3,
        dropout=0,
        with_noise_level_emb=True,
        use_affine_level=False,
        image_size=128,
        num_classes=None,
        additive_class_emb=False,
        use_nd_dropout=False,
        T=None,
        use_second_time=False,
        no_mid_attn=False,
        mid_cattn=False,
        output_residual=False,
    ):
        super().__init__()
        self.dims = dims
        self.image_size = image_size
        self.use_second_time = use_second_time
        self.output_residual = output_residual

        # if with_noise_level_emb:
        #     noise_level_channel = inner_channel * 4
        #     self.noise_level_mlp = nn.Sequential(
        #         PositionalEncoding(inner_channel),
        #         nn.Linear(inner_channel, inner_channel * 4),
        #         Swish(),
        #         nn.Linear(inner_channel * 4, noise_level_channel),
        #     )
        # else:
        #     assert False
        #     noise_level_channel = None
        #     self.noise_level_mlp = None

        # self.num_classes = num_classes
        # if num_classes is not None:
        #     noise_level_channel = 4 * inner_channel
        #     self.class_embedding = ClassEmbedding(num_classes, inner_channel * 4, noise_level_channel)

        if with_noise_level_emb:
            noise_level_channel = inner_channel * 4
            if T is None:
                self.noise_level_mlp = nn.Sequential(
                    PositionalEncoding(inner_channel),
                    nn.Linear(inner_channel, inner_channel * 2),
                    Swish(),
                    nn.Linear(inner_channel * 2, noise_level_channel),
                )
            else:
                self.noise_level_mlp = DiscreteTimeEmbedding(T, inner_channel * 2, noise_level_channel)
            if use_second_time:
                self.noise_level_mlp2 = deepcopy(self.noise_level_mlp)
        else:
            assert False
            noise_level_channel = None
            self.noise_level_mlp = None

        self.additive_class_emb = additive_class_emb
        self.num_classes = num_classes
        if num_classes is not None:
            self.class_embedding = ClassEmbedding(num_classes, inner_channel * 2, noise_level_channel)
            if not additive_class_emb:
                noise_level_channel *= 2

        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size
        downs = [conv_nd(dims, in_channel, inner_channel, kernel_size=3, padding=1)]
        for ind in range(num_mults):
            is_last = ind == num_mults - 1
            use_attn = now_res in attn_res
            use_cattn = now_res in cattn_res
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(
                    ResnetBlocWithAttn(
                        dims,
                        pre_channel,
                        channel_mult,
                        noise_level_emb_dim=noise_level_channel,
                        norm_groups=norm_groups,
                        dropout=dropout,
                        with_attn=use_attn,
                        with_cattn=use_cattn,
                        use_affine_level=use_affine_level,
                        use_nd_dropout=use_nd_dropout,
                    )
                )
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(dims, pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res // 2
        self.downs = nn.ModuleList(downs)

        self.mid = nn.ModuleList(
            [
                ResnetBlocWithAttn(
                    dims,
                    pre_channel,
                    pre_channel,
                    noise_level_emb_dim=noise_level_channel,
                    norm_groups=norm_groups,
                    dropout=dropout,
                    with_attn=not no_mid_attn,
                    with_cattn=mid_cattn,
                    use_affine_level=use_affine_level,
                    use_nd_dropout=use_nd_dropout,
                ),
                ResnetBlocWithAttn(
                    dims,
                    pre_channel,
                    pre_channel,
                    noise_level_emb_dim=noise_level_channel,
                    norm_groups=norm_groups,
                    dropout=dropout,
                    with_attn=False,
                    use_affine_level=use_affine_level,
                    use_nd_dropout=use_nd_dropout,
                ),
            ]
        )

        ups = []
        for ind in reversed(range(num_mults)):
            is_last = ind < 1
            use_attn = now_res in attn_res
            use_cattn = now_res in cattn_res
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks + 1):
                ups.append(
                    ResnetBlocWithAttn(
                        dims,
                        pre_channel + feat_channels.pop(),
                        channel_mult,
                        noise_level_emb_dim=noise_level_channel,
                        norm_groups=norm_groups,
                        dropout=dropout,
                        with_attn=use_attn,
                        with_cattn=use_cattn,
                        use_nd_dropout=use_nd_dropout,
                    )
                )
                pre_channel = channel_mult
            if not is_last:
                ups.append(Upsample(dims, pre_channel))
                now_res = now_res * 2

        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(dims, pre_channel, default(out_channel, in_channel), groups=norm_groups)

    def forward(self, x, t, c=None, s=None):
        t = self.noise_level_mlp(t) if exists(self.noise_level_mlp) else None
        if self.use_second_time:
            t = t + self.noise_level_mlp2(s)
        if self.num_classes is not None:
            if self.additive_class_emb:
                t = t + self.class_embedding(c)
            else:
                t = th.cat([t, self.class_embedding(c)], dim=1)

        x_org = x
        feats = []
        for layer in self.downs:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)
            feats.append(x)

        for layer in self.mid:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)

        for layer in self.ups:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(th.cat((x, feats.pop()), dim=1), t)
            else:
                x = layer(x)

        x = self.final_conv(x)
        if self.output_residual:
            x = x + x_org[:, : x.size(1)]
        return x
