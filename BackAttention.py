import torch
import torch.nn as nn
import torch.nn.functional as F
from ConvertD import ConvertD,RConvertD


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.linear1 = nn.Linear(in_channels, in_channels // reduction, bias=False)

        self.bn = nn.BatchNorm1d(in_channels // reduction)

        self.act = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(in_channels // reduction, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, _, _ = x.shape
        y = self.avg_pool(x).reshape(B, C)
        y = self.linear1(y)
        y = self.bn(y)
        y = self.act(y)
        y = self.linear2(y)
        y = self.sigmoid(y).view(B, C, 1, 1)
        return x * y


class SpatialAttention(nn.Module):
    def __init__(self, in_channels, height, width, num_heads=8):
        super(SpatialAttention, self).__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.scale = 1.0 / (self.head_dim ** 0.5)

        self.out_proj= nn.Linear(in_channels, in_channels, bias=False)
        self.qkv = nn.Linear(in_channels, in_channels * 3, bias=False)

        self.pos_bias = nn.Parameter(torch.zeros(self.num_heads, height * width, height * width))
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(in_channels)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H * W == self.pos_bias.size(1) and H * W == self.pos_bias.size(2), \
            f"Input spatial size {H * W} doesn't match pos_bias size {self.pos_bias.shape}"
        x_flat = x.view(B, C, -1).transpose(1, 2)
        qkv = self.qkv(x_flat).chunk(3, dim=-1)
        q, k, v = [t.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2) for t in qkv]
        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = F.softmax(attn + self.pos_bias.unsqueeze(0), dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, -1, C)
        out = self.out_proj(out)
        out = self.dropout(out)

        out = out + x_flat
        out = self.norm(out).transpose(1, 2).view(B, C, H, W)

        return out


class CrossAttention(nn.Module):
    def __init__(self, in_channels, num_heads = 8):
        super(CrossAttention, self).__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        assert in_channels % num_heads == 0, f"dim {in_channels} must be divisible by num_heads {num_heads}"
        self.scale = 1.0 / (self.head_dim ** 0.5)

        self.q_proj = nn.Linear(in_channels, in_channels)
        self.k_proj = nn.Linear(in_channels, in_channels)
        self.v_proj = nn.Linear(in_channels, in_channels)
        self.out_proj = nn.Linear(in_channels, in_channels)
        self.ln = nn.LayerNorm(in_channels)

    def forward(self, q_feat, kv_feat):
        B, C, H, W = q_feat.shape
        q = self.q_proj(q_feat.flatten(2).transpose(1, 2)).view(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv_feat.flatten(2).transpose(1, 2)).view(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_feat.flatten(2).transpose(1, 2)).view(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = self.ln(self.out_proj(out.view(B,H*W,C))).transpose(1, 2).view(B, C, H, W)
        return out + kv_feat


class BackAttentionFusion(nn.Module):
    def __init__(self, in_channels, height, width):
        super(BackAttentionFusion, self).__init__()
        self.channel_attn = ChannelAttention(in_channels)
        self.spatial_attn = SpatialAttention(in_channels, height, width)
        self.cross_attn = CrossAttention(in_channels)

    def forward(self, Back, Backbone):
        Back = ConvertD(Back)
        Backbone = ConvertD(Backbone)
        Backfeat = self.channel_attn(Back)
        Backfeat = self.spatial_attn(Backfeat)
        out = self.cross_attn(Backbone, Backfeat)
        out = RConvertD(out)
        return out
