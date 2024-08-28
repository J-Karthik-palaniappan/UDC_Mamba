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

# Example usage:
# img1 and img2 should be torch tensors of shape [1, C, H, W] and have the same dtype and range.
# For example, if your images are in [0, 1] range, use them directly.
if __name__ == "__main__":
    img1 = torch.randn(3, 256, 256)  # Replace with your actual image tensor
    img2 = torch.randn(3, 256, 256)  # Replace with your actual image tensor

    ssim_value = calculate_ssim(img1, img2)
    print(f"SSIM: {ssim_value.item()}")