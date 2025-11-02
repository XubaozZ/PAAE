from models.paae import *


def get_important_patch_coordinates_from_attn(
		attn_map, window_size, shift_size,
		B, H, W, isdown=True, topk_windows=1, topk_patches=8
):

	Wh = Ww = window_size
	Sh = Sw = shift_size

	if H > Wh:
		N = Wh * Ww
	else:
		N = H * W
	num_windows_per_img = attn_map.shape[0] // B

	num_win_h = (H + Sh) // Wh
	num_win_w = (W + Sw) // Ww
	assert num_windows_per_img == num_win_h * num_win_w

	attn_scores = attn_map.mean(dim=1)  # (B_, N, N) N=256
	patch_scores = attn_scores.mean(dim=1)  # (B_, N) 32*1,256

	patch_scores = patch_scores.view(B, num_windows_per_img, N)  # (B, num_win, N)
	topk_window_ids = patch_scores.mean(dim=2).topk(topk_windows, dim=1).indices  # (B, topk_windows)

	coords = []

	for b in range(B):
		for win_rank, win_id in enumerate(topk_window_ids[b]):
			win_id = win_id.item()
			win_score = patch_scores[b, win_id]  # shape: (N,)

			if topk_windows == 1 or win_rank < topk_windows // 4:
				num_patches = topk_patches
			elif win_rank < topk_windows // 2:
				num_patches = max(1, topk_patches // 2)
			else:
				num_patches = max(1, topk_patches // 4)

			topk_patch_ids = win_score.topk(num_patches).indices  # shape: (num_patches,)

			win_h_idx = win_id // num_win_w
			win_w_idx = win_id % num_win_w
			start_h = win_h_idx * Wh
			start_w = win_w_idx * Ww

			for pid in topk_patch_ids:
				pid = pid.item()
				patch_h, patch_w = divmod(pid, Ww)
				abs_h = start_h + patch_h
				abs_w = start_w + patch_w
				orig_h = (abs_h + Sh) % H
				orig_w = (abs_w + Sw) % W

				if isdown:
					merged_h = orig_h // 2
					merged_w = orig_w // 2
				else:
					merged_h = orig_h
					merged_w = orig_w
				coords.append((b, merged_h, merged_w))

	return coords

def apply_patch_mask(feat, selected_coords):

	feat = ConvertD(feat)
	B, C, H, W = feat.shape
	device = feat.device

	mask = torch.zeros((B, C, H, W), dtype=feat.dtype, device=device)
	for b, h, w in selected_coords:
		if 0 <= b < B and 0 <= h < H and 0 <= w < W:
			mask[b, :, h, w] = 1.0
	masked_feat = feat * mask

	masked_feat = RConvertD(masked_feat)  # B,L,C

	return masked_feat

class FeatureFusion(nn.Module):
    def __init__(self):
        super().__init__()

        self.reduce_conv = nn.ModuleList([
            nn.Linear(1024, 512),
            nn.Linear(512, 256),
            nn.Linear(256, 128),
        ])

        self.upsample = nn.ModuleList([
            UpsampleConv(1024, 512),
            UpsampleConv(512, 256),
            UpsampleConv(256, 128)
        ])

        self.displacement = nn.ModuleList([
            LearnableDisplacement(24, 24),
            LearnableDisplacement(48, 48),
            LearnableDisplacement(96, 96),
        ])

        self.bias_template_shapes = [(576, 512), (2304, 256), (9216, 128)]
        self.bias_templates = nn.ParameterList([
            nn.Parameter(torch.zeros(1, s[0], s[1])) for s in self.bias_template_shapes
        ])

        self.feature_dims = [[16, 9216, 128], [16, 2304, 256], [16, 576, 512], [16, 144, 1024]]  # base384
        self.feature_dims4 = [[16, 128, 96, 96], [16, 256, 48, 48], [16, 512, 24, 24], [16, 1024, 12, 12]]  # base384
        self.feature_dims4ori = [[16, 128, 96, 96], [16, 256, 48, 48], [16, 512, 24, 24], [16, 1024, 12, 12]]  # base384
        self.norms = nn.ModuleList([nn.LayerNorm(self.feature_dims[i][2]) for i in range(4)])

        target_probs = torch.tensor([0.1, 0.1, 0.1, 0.7])
        init_logits = torch.log(target_probs + 1e-8)
        self.paralist = nn.Parameter(init_logits.clone().detach())

    def forward(self, xlist,batch_size = 16,windowsize = 7,shiftsize=4,attnmaplist = None):
        assert len(xlist) == 4, "Expected xlist of length 4"
        xlist[0] = apply_patch_mask(xlist[0],get_important_patch_coordinates_from_attn(attnmaplist[0][-1], windowsize, shiftsize,
        																   batch_size,
        																   self.feature_dims4ori[0][2],
        																   self.feature_dims4ori[0][3], isdown=True,
        																   topk_windows=8,
        																   topk_patches=64))
        xlist[1] = apply_patch_mask(xlist[1],
        						 get_important_patch_coordinates_from_attn(attnmaplist[1][-1], windowsize, shiftsize,
        																   batch_size,
        																   self.feature_dims4ori[1][2],
        																   self.feature_dims4ori[1][3], isdown=True,
        																   topk_windows=4,
        																   topk_patches=64))
        xlist[2] = apply_patch_mask(xlist[2],
        						 get_important_patch_coordinates_from_attn(attnmaplist[2][-1], windowsize, shiftsize,
        																   batch_size,
        																   self.feature_dims4ori[2][2],
        																   self.feature_dims4ori[2][3], isdown=True,
        																   topk_windows=1,
        																   topk_patches=64))
        xlist[3] = apply_patch_mask(xlist[3],
        						 get_important_patch_coordinates_from_attn(attnmaplist[3][-1], windowsize, shiftsize,
        																   batch_size,
        																   self.feature_dims4ori[3][2],
        																   self.feature_dims4ori[3][3], isdown=False,
        																   topk_windows=1,
        																   topk_patches=64))

        B = xlist[0].size(0)
        for i in range(3, 0, -1):  # 3→2→1
            H, W = {3: (24, 24), 2: (48, 48), 1: (96, 96)}[i]
            upsampled = self.upsample[3 - i](xlist[i])  # [B, H*W, C]
            C = upsampled.shape[-1]

            upsampled = upsampled .transpose(1, 2).reshape(B, C, H, W)
            upsampled = self.displacement[3 - i](upsampled)
            upsampled = upsampled.flatten(2).transpose(1, 2)  # [B, H*W, C]
            xlist[i - 1] = xlist[i - 1] + upsampled

        return xlist



class LearnableDisplacement(nn.Module):
    def __init__(self, h, w):
        super().__init__()
        self.h, self.w = h, w
        self.offset = nn.Parameter(torch.zeros(1, h, w, 2))
        nn.init.uniform_(self.offset, a=-0.02, b=0.02)

    def forward(self, feat):
        B, C, H, W = feat.shape
        device = feat.device

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij'
        )
        base_grid = torch.stack((grid_x, grid_y), dim=-1)  # [H, W, 2]
        base_grid = base_grid.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, W, 2]

        sampling_grid = base_grid + self.offset  # [B, H, W, 2]
        sampling_grid = sampling_grid.clamp(-1, 1)

        sampled = F.grid_sample(feat, sampling_grid, mode='bilinear', align_corners=True)
        return sampled


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
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

class ConvRefiner(nn.Module):
    def __init__(self, channels, drop_path=0.1, layer_scale_init_value=1e-6):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((channels)), requires_grad=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        residual = self.conv(x)
        residual = self.layer_scale.view(1, -1, 1, 1) * residual  # LayerScale
        return x + self.drop_path(residual)  # Residual + DropPath


class UpsampleConv(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = ConvertD(x)
        x = self.upsample(x)
        x = self.conv(x)
        return RConvertD(x)


class DownConvRefiner(nn.Module):
    def __init__(self, in_channels, drop_path=0.1, layer_scale_init_value=1e-6, downsample=True):
        super().__init__()
        self.downsample = downsample

        mid_channels = in_channels * 2 if downsample else in_channels
        self.out_channels = mid_channels

        self.down = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        ) if downsample else nn.Identity()

        self.conv = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
        )

        self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((mid_channels)), requires_grad=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = ConvertD(x)
        x = self.down(x)  # optional downsampling
        residual = self.conv(x)
        residual = self.layer_scale.view(1, -1, 1, 1) * residual
        return RConvertD(x + self.drop_path(residual))

