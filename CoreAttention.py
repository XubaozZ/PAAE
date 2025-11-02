import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ConvertD import ConvertD,RConvertD


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # (B, 1, 1, 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        binary_tensor = torch.floor(random_tensor)
        return x.div(keep_prob) * binary_tensor

class ChannelEnhancement(nn.Module):
    def __init__(self, in_channels):
        super(ChannelEnhancement, self).__init__()
        self.in_channels = in_channels
        self.half_channels = in_channels // 2

        self.path1_pwc = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1)
        self.path1_dwc = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, padding=1, groups=self.half_channels)

        self.path2_pwc = nn.Conv2d(self.half_channels, self.half_channels, kernel_size=1)
        self.path2_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        )
        self.norm = nn.BatchNorm2d(in_channels)
        self.act = nn.ReLU(inplace=True)
        self.droppath = DropPath()

    def forward(self, x):
        B, C, H, W = x.shape
        x_res = x
        gap = F.adaptive_avg_pool2d(x, 1)  # (B, C, 1, 1)
        gate = torch.sigmoid(gap)  # (B, C, 1, 1)

        gate_flat = gate.view(B, C)
        topk_vals, topk_idx = torch.topk(gate_flat, self.half_channels, dim=1)

        mask = torch.zeros_like(gate_flat)
        mask.scatter_(1, topk_idx, 1)
        mask = mask.view(B, C, 1, 1)#(B, C, 1, 1)

        x1 = x * mask  # B,C,H,W
        x2 = x * (1 - mask)  #  B,C,H,W

        x1 = self.path1_pwc(x1)#B,C,H,W
        x1 = self.path1_dwc(x1)#B,C,H,W

        x2_half = self.path2_pwc(x2[:, :self.half_channels])#B,C/2,h,w
        x2_res = x2[:, self.half_channels:]#B,C/2,h,w
        x2 = torch.cat([x2_half, x2_res], dim=1)#B,C,H,W

        out = x1 + x2  # 形状为B,C,H,W
        out = self.norm(out)
        out = self.act(out)
        out = self.droppath(out)
        out = out + x_res
        return out


class SpatialEnhancement(nn.Module):
    def __init__(self, in_channels):
        super(SpatialEnhancement, self).__init__()
        self.pwc = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.branch3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.branch5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels)
        self.branch7 = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels)
        self.fuse3 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.fuse5 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.fuse7 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.norm = nn.BatchNorm2d(in_channels)
        self.act = nn.ReLU(inplace=True)
        self.droppath = DropPath()
    def forward(self, x):
        x_res = x
        x = self.pwc(x)

        x3 = self.fuse3(self.branch3(x))
        x5 = self.fuse5(self.branch5(x))
        x7 = self.fuse7(self.branch7(x))
        x = x3 + x5 + x7

        x = self.norm(x)
        x = self.act(x)
        x = self.droppath(x)
        x = x + x_res

        return x


class BiCrossAttention(nn.Module):
    def __init__(self, in_channels ,num_heads = 16,isDown = False):
        super(BiCrossAttention, self).__init__()
        assert in_channels % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = in_channels
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.q2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.gate_q = nn.Parameter(torch.randn(num_heads, in_channels) * 0.02)
        self.gate_k = nn.Parameter(torch.randn(num_heads, in_channels) * 0.02)
        self.gate_v = nn.Parameter(torch.randn(num_heads, in_channels) * 0.02)

        self.reduce_channels = nn.Linear(in_channels, in_channels//num_heads)

        self.dropout = nn.Dropout(0.2)
        self.attndropout = nn.Dropout(0.1)

        target_probs1 = torch.tensor([0.5, 0.5])
        init_logits1 = torch.log(target_probs1 + 1e-8)
        self.alpha = nn.Parameter(init_logits1.clone().detach())

        self.out = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        target_probs2 = torch.tensor([0.1, 0.9])
        init_logits2 = torch.log(target_probs2 + 1e-8)
        self.paralist = nn.Parameter(init_logits2.clone().detach())

    def apply_global_gating(self, x, gate_weights):
        B, C, H, W = x.shape
        x = x.unsqueeze(1)  # (B, 1, C, H, W)
        gates = gate_weights.view(1, self.num_heads, C, 1, 1)  # (1, H, C, 1, 1)
        x_gates = (x * gates).permute(0,1,3,4,2)
        x_gates = self.reduce_channels(x_gates)
        return x_gates.permute(0,1,4,2,3)  # (B, Heads, C/heads, H, W)

    def compute_head_diversity_loss(self, gated_q):
        # gated_q: (B, H, C, H, W)
        B, H, C, _, _ = gated_q.shape
        x = gated_q.mean(dim=[3, 4])  # (B, H, C)

        x = F.normalize(x, p=2, dim=-1)

        loss = 0.0
        count = 0
        for i in range(H):
            for j in range(i + 1, H):
                sim = (x[:, i] * x[:, j]).sum(dim=-1)  # (B,)
                loss += sim.mean()
                count += 1
        if count > 0:
            loss = loss / count
        return 1 - loss  # 1 - mean cosine similarity

    def forward(self, x1, x2):
        assert x1.shape == x2.shape, "x1 and x2 must have the same shape"
        B, C, H, W = x1.shape

        q1 = self.q1(x1)# B, C, H, W
        k1 = self.k1(x2)
        v1 = self.v1(x2)

        q2 = self.q2(x2)
        k2 = self.k2(x1)
        v2 = self.v2(x1)

        gated_q1 = self.apply_global_gating(q1, self.gate_q)## (B, Hads, C, H, W)
        gated_k1 = self.apply_global_gating(k1, self.gate_k)
        gated_v1 = self.apply_global_gating(v1, self.gate_v)

        gated_q2 = self.apply_global_gating(q2, self.gate_q)
        gated_k2 = self.apply_global_gating(k2, self.gate_k)
        gated_v2 = self.apply_global_gating(v2, self.gate_v)

        def attn(q, k, v):
            B, H, C, Hh, Ww = q.shape# (B, Heads, C/heads, H, W)
            q = q.view(B, H, C, Hh * Ww).permute(0, 1, 3, 2)  # B, H, HW, C
            k = k.view(B, H, C, Hh * Ww)                     # B, H, C, HW
            v = v.view(B, H, C, Hh * Ww).permute(0, 1, 3, 2)  # B, H, HW, C

            attn = torch.matmul(q, k) * self.scale  # B, H, HW, HW
            attn = F.softmax(attn, dim=-1)

            attn = self.attndropout(attn)

            out = torch.matmul(attn, v)  # B, H, HW, C
            out = out.permute(0, 1, 3, 2).contiguous().view(B, self.num_heads * self.head_dim, Hh, Ww)
            return out

        out1 = attn(gated_q1, gated_k1, gated_v1)
        out2 = attn(gated_q2, gated_k2, gated_v2)

        weights1 = F.softmax(self.alpha, dim=0)
        fused = weights1[0] * out1 + weights1[1] * out2

        weights2 = F.softmax(self.paralist, dim=0)
        out = weights2[0] * self.out(self.dropout(fused)) + weights2[1] * x2

        diversity_loss = self.compute_head_diversity_loss(gated_q1) + self.compute_head_diversity_loss(gated_q2)

        return out, diversity_loss



class CoreFeatureEnhancer(nn.Module):
    def __init__(self, in_channels,num_heads=16,isDown = True):
        super(CoreFeatureEnhancer, self).__init__()
        self.num_heads = num_heads
        self.idDown = isDown
        self.channel_enhance = ChannelEnhancement(in_channels)
        self.spatial_enhance = SpatialEnhancement(in_channels)
        self.cross_attn = BiCrossAttention(in_channels,num_heads=self.num_heads,isDown = self.idDown)

    def forward(self, Core, Backbone):
        Core = ConvertD(Core)
        Backbone = ConvertD(Backbone)
        Corefeat = self.channel_enhance(Core)
        Corefeat = self.spatial_enhance(Corefeat)

        out, diversity_loss = self.cross_attn(Corefeat, Backbone)
        out = RConvertD(out)
        return out, diversity_loss
