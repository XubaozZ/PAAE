import torch
import torch.nn as nn

class ReversePatchMerging(nn.Module):
    def __init__(self, output_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.output_resolution = output_resolution


        self.dim = dim

        self.recover = nn.Linear(dim, dim * 2, bias=False)

        self.norm = norm_layer(dim * 2)


    def forward(self, x):
        H, W = self.output_resolution
        B, L, C2 = x.shape

        assert C2 == self.dim, "channel mismatch"

        H_half, W_half = H // 2, W // 2
        assert L == H_half * W_half, "length mismatch with expected half resolution"

        if self.recover.weight.device != x.device:
            self.recover = self.recover.to(x.device)
        x = self.recover(x)
        if self.norm.weight.device != x.device:
            self.norm = self.norm.to(x.device)
        x = self.norm(x)

        x = x.view(B, H_half, W_half, 4 * (C2 // 2))
        C = C2 // 2


        x0 = x[:, :, :, 0*C:1*C]
        x1 = x[:, :, :, 1*C:2*C]
        x2 = x[:, :, :, 2*C:3*C]
        x3 = x[:, :, :, 3*C:4*C]


        out = torch.zeros(B, H, W, C, dtype=x.dtype, device=x.device)

        out[:, 0::2, 0::2, :] = x0
        out[:, 1::2, 0::2, :] = x1
        out[:, 0::2, 1::2, :] = x2
        out[:, 1::2, 1::2, :] = x3

        out = out.view(B, H * W, C)
        return out
