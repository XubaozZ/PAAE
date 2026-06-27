import torch
from ConvertD import ConvertD,RConvertD

def extract_core_and_background(images, coordinates):
    images = ConvertD(images)
    B, C, H, W = images.shape
    core_images = torch.zeros_like(images)
    background_images = images.clone()

    for i in range(B):
        x1, y1, x2, y2 = coordinates[i]
        core_images[i, :, y1:y2, x1:x2] = images[i, :, y1:y2, x1:x2]
        background_images[i, :, y1:y2, x1:x2] = 0

    core_images = RConvertD(core_images)
    background_images = RConvertD(background_images)
    return core_images, background_images
