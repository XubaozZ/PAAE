from Featurefusion import *
from Coordinate import *

from models.backbone.Swin_Transformer import swin_backbone, PatchMerging, Mlp,swin_backbone_tiny


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

class FeatureEnhancer(nn.Module):
    def __init__(self):
        super().__init__()

        self.feature_dims = [[16, 9216, 128], [16, 2304, 256], [16, 576, 512], [16, 144, 1024]]  # base384
        self.feature_dims4 = [[16, 128, 96, 96], [16, 256, 48, 48], [16, 512, 24, 24], [16, 1024, 12, 12]]  # base384
        self.feature_dims4ori = [[16, 128, 96, 96], [16, 256, 48, 48], [16, 512, 24, 24], [16, 1024, 12, 12]]  # base384


        self.featurefusion = FeatureFusion().to(device='cuda')
        self.merging_patch0 = PatchMerging(input_resolution=(self.feature_dims4[0][2], self.feature_dims4[0][3]),
                                      dim=self.feature_dims4[0][1])
        self.merging_patch1 = PatchMerging(input_resolution=(self.feature_dims4[1][2], self.feature_dims4[1][3]),
                                      dim=self.feature_dims4[1][1])
        self.merging_patch2 = PatchMerging(input_resolution=(self.feature_dims4[2][2], self.feature_dims4[2][3]),
										   dim=self.feature_dims4[2][1])
        self.down1 = DownConvRefiner(128)
        self.down2 = DownConvRefiner(256)
        self.down3 = DownConvRefiner(512)

        self.remerging_patch2 = ReversePatchMerging(output_resolution = (self.feature_dims4[2][2],self.feature_dims4[2][3]), dim = self.feature_dims4[3][1], norm_layer=nn.LayerNorm)
        self.Corefeatureenhancer2 = CoreFeatureEnhancer(self.feature_dims4[2][1],num_heads=16)
        self.Corefeatureenhancer3 = CoreFeatureEnhancer(self.feature_dims4[3][1],num_heads=32,isDown=False)

        target_probs = torch.tensor([0.5, 0.5])
        init_logits = torch.log(target_probs + 1e-8)
        self.paralist = nn.Parameter(init_logits.clone().detach())


    def forward(self, x_layers,batch_size,windowsize,shiftsize,attnmaplist=None):
        x_fusion = self.featurefusion(x_layers,batch_size,windowsize,shiftsize,attnmaplist)

        Core0 = x_fusion[1] + self.down1(x_fusion[0])

        Core0 = x_fusion[2] + self.down2(Core0)

        Core0, diversity_loss1 = self.Corefeatureenhancer2(Core0, x_fusion[2])

        Core0 = x_fusion[3] + self.down3(Core0)

        Core3, diversity_loss2 = self.Corefeatureenhancer3(Core0, x_fusion[3])

        diversity_loss = diversity_loss2+diversity_loss1

        return Core3, diversity_loss



class paae(nn.Module):
	def __init__(self, dim, input_size, backbone=None, parts_ratio=2, num_heads=16,
	            feature_weights_pooling=True, att_drop=0.2, head_drop=0.5,
	             parts_drop=0.2, num_classes=200, pos=True, parts_base=0., cross_layer=False,
	             label_smooth=0.0, mixup=0.,backbone_type='hier'):
		super(paae, self).__init__()
		self.num_heads = num_heads
		self.parts_ratio = parts_ratio
		self.input_size = (input_size//32,input_size//32) if backbone_type=='hier' else (input_size//16,input_size//16)
		self.dim = dim
		self.num_classes = num_classes
		self.fwp = feature_weights_pooling
		self.mixup = mixup
		self.ce = LabelSmoothingCrossEntropy(smoothing=label_smooth)
		self.ls = nn.LogSoftmax(dim=-1)
		self.activation = nn.GELU()
		self.dis = nn.KLDivLoss(reduction='batchmean',log_target=True)
		self.dis = nn.PairwiseDistance(2,1e-8)

		self.backbone_type = backbone_type
		if self.fwp:
			self.pooling = self.feature_weights_pooling
			stage_weights = 4 if cross_layer else 1
			self.pooling_weights = nn.Parameter(torch.ones(stage_weights, 1, 1, 1) / stage_weights)
		else:
			self.pooling = nn.AdaptiveAvgPool1d(1)
			self.conv = nn.Conv2d(2,1,3,1,1)
		self.featureenhence = FeatureEnhancer()

		self.norm = nn.LayerNorm(dim)
		self.embed_dim = 96
		self.num_features = 1024
		self.avgpool = nn.AdaptiveAvgPool1d(1)
		self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

		self.head_drop = nn.Dropout(0.2)
		self.show = nn.Identity()
		self.apply(self.init_weights)
		self.backbone = backbone
		self.assess = False
		self.save_feature = None
		self.count = 0



	def feature_weights_pooling(self, x, feature_weights):
		if self.assess:
			os.makedirs(f'../visualize/feature_weights/', exist_ok=True)
			torch.save(x, '../visualize/feature_weights/x.pt')
			torch.save(feature_weights, '../visualize/feature_weights/weights.pt')
		sum_feature_weights = (feature_weights * self.pooling_weights).sum(0)
		sum_feature_weights = sum_feature_weights / sum_feature_weights.sum(-2).unsqueeze(-1)
		x = x @ sum_feature_weights
		return x, sum_feature_weights

	def init_weights(self, m):
		if isinstance(m, (nn.Linear, nn.Conv2d)):
			nn.init.kaiming_normal_(m.weight)
			if isinstance(m, nn.Linear) and m.bias is not None:
				nn.init.constant_(m.bias, 0)
		elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
			nn.init.constant_(m.bias, 0)
			nn.init.constant_(m.weight, 1.0)
		nn.init.constant_(self.head.weight, 0)
		nn.init.constant_(self.head.bias, 0)

	def normalize_cam(self,grad):
		min,_ = grad.min(-1, keepdim=True)
		grad = grad - min
		grad = grad / (1e-8 + grad.sum(-1, keepdim=True))
		return grad

	def flops(self):
		flops = 0
		flops += self.backbone.flops()
		# HPR
		flops += self.block.flops()
		# Delete Original Norm
		flops -= self.dim * self.input_size[0] * self.input_size[0]
		# Delete Original Head
		flops -= self.dim * self.input_size[0] * self.input_size[0]
		# Norm
		flops += self.dim * 15 / 8 * self.input_size[0] * self.input_size[0]
		# Multi-Grained Fusion
		flops += self.dim * self.input_size[0] * self.input_size[0]
		# Head
		flops += self.dim * 15 / 8 * self.num_classes
		return flops


	def forward(self, x, label=None,epoch=None):
		x, attnmaplist = self.backbone(x)

		output_dir = config.data.log_path if hasattr(config, "data") else "./logs"
		os.makedirs(output_dir, exist_ok=True)
		if epoch is None:
			epoch = -1
		save_path = os.path.join(output_dir, f"feature_stats_epoch_{epoch + 1}.csv")
		with open(save_path, 'w') as f:
			f.write("Stage,Mean,Std\n")

		for i, feat in enumerate(x):
			mean = feat.mean().item()
			std = feat.std().item()

			with open(save_path, 'a') as f:
				f.write(f"{i + 1},{mean:.6f},{std:.6f}\n")

		batch_size = x[-1].shape[0]
		Core3, diversity_loss = self.featureenhence(x, batch_size, 12, 12//2, attnmaplist)
		x = self.norm(Core3)
		x = self.head_drop(x)
		x = self.avgpool(x.transpose(1, 2))  # B C 1
		x = torch.flatten(x, 1)
		x = self.head(x)

		if self.training and label is not None and self.mixup <= 0:
			loss_ce, loss_cam=0, 0
			label = label.long()
			loss_ce = self.ce(x, label)
			loss = [loss_ce + loss_cam , loss_ce, loss_cam]
			return x, loss
		else:
			return x

if __name__ == '__main__':
	dim = 1024
	img_size = 384
	batch = 2
	x = torch.rand(batch, 3, img_size, img_size)
	label = torch.randint(200,(batch,))
	backbone = swin_backbone(window_size=img_size//32, img_size=img_size, num_classes=200, cross_layer=True)
	paae = paae(dim, img_size, backbone,
	                         parts_drop=0.2, parts_base=0., cross_layer=True, feature_weights_pooling=False)

	from thop import profile

	print('Backbone FLOPs = ' + str(paae.backbone.flops() / 1000 ** 3) + 'G')
	print('Backbone Params = ' + str(count_parameters(paae.backbone)) + 'M')
	y, loss = paae(x, label)
	print('Ours FLOPs = ' + str(paae.flops() / 1000 ** 3) + 'G')
	print('Ours Params = ' + str(count_parameters(paae)) + 'M')
	print(loss)

