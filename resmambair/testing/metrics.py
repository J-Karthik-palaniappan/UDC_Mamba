import torch
from skimage.metrics import structural_similarity as ssim
import torch.nn.functional as F
import numpy as np

# Function to calculate PSNR
def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr.item()

def calculate_ssim(img1, img2, win_size=7):
    img1 = img1.permute(1, 2, 0).cpu().numpy()
    img2 = img2.permute(1, 2, 0).cpu().numpy()
    # Ensure win_size is suitable for the image size
    min_side = min(img1.shape[0], img1.shape[1])
    win_size = min(win_size, min_side)
    win_size = win_size if win_size % 2 == 1 else win_size - 1  # Ensure win_size is odd
    return ssim(img1, img2, multichannel=True, data_range=img2.max() - img2.min(), win_size=win_size)

# def gaussian(window_size, sigma):
#     gauss = torch.Tensor([np.exp(-(x - window_size//2)*2/float(2*sigma*2)) for x in range(window_size)])
#     return gauss/gauss.sum()

# def create_window(window_size, channel):
#     _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
#     _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
#     window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
#     return window

# def calculate_ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
#     if val_range is None:
#         max_val = 1 if torch.max(img1) <= 1 else 255
#         min_val = 0 if torch.min(img1) >= 0 else -1

#         L = max_val - min_val
#     else:
#         L = val_range

#     padd = 0
#     (_, channel, height, width) = img1.size()
#     if window is None:
#         real_size = min(window_size, height, width)
#         window = create_window(real_size, channel).to(img1.device)

#     mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
#     mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

#     mu1_sq = mu1.pow(2)
#     mu2_sq = mu2.pow(2)
#     mu1_mu2 = mu1 * mu2

#     sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
#     sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
#     sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

#     C1 = (0.01 * L) ** 2
#     C2 = (0.03 * L) ** 2

#     ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

#     if size_average:
#         return ssim_map.mean()
#     else:
#         return ssim_map.mean(1).mean(1).mean(1)

# Example usage:
# img1 and img2 should be torch tensors of shape [1, C, H, W] and have the same dtype and range.
# For example, if your images are in [0, 1] range, use them directly.
if __name__ == "__main__":
    img1 = torch.randn(3, 256, 256)  # Replace with your actual image tensor
    img2 = torch.randn(3, 256, 256)  # Replace with your actual image tensor

    ssim_value = calculate_ssim(img1, img2)
    print(f"SSIM: {ssim_value.item()}")