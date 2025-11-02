
def ConvertD(images):
    B,L,C = images.shape
    H = W = int(L ** 0.5)
    assert H * W == L, "L must be a perfect square"
    images = images.view(B, H, W, C).permute(0, 3, 1, 2)

    return images

def RConvertD(images):
    B, C, H, W = images.shape
    images = images.permute(0, 2, 3, 1).view(B, H * W, C)

    return images



