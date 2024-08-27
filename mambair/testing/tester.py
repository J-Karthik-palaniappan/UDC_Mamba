import os
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from testing.metrics import calculate_psnr, calculate_ssim
from mambair.mambairunet_arch import MambaIRUNet

# Setup CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Function to load the model
def load_model(checkpoint_path):
    model = MambaIRUNet()
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    # model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model

# Define image transformations
def get_transform(imageNet = True):
    if imageNet: 
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else: 
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

def get_inv_transform(imageNet=True):
    if imageNet:
        return transforms.Compose([
            transforms.Normalize(mean=[0., 0., 0.],
                                 std=[1/0.229, 1/0.224, 1/0.225]),
            transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                 std=[1., 1., 1.]),
        ])
    else:
        return transforms.Compose([
            transforms.Normalize(mean=[0., 0., 0.],
                                 std=[2, 2, 2]),
            transforms.Normalize(mean=[-0.5, -0.5, -0.5],
                                 std=[1., 1., 1.]),
        ])

# Function to process a single image
def process_image(image_path, expected_path, model, transform, inv_transform, patch_size=128):
    image = Image.open(image_path).convert('RGB')
    expected_image = Image.open(expected_path).convert('RGB')
    
    input_tensor = transform(image).unsqueeze(0).to(device)
    expected_tensor = transform(expected_image).unsqueeze(0).to(device)

    _, _, height, width = input_tensor.shape

    # Calculate the number of patches needed
    h_patches = (height + patch_size - 1) // patch_size
    w_patches = (width + patch_size - 1) // patch_size

    restored_tensor = torch.zeros_like(input_tensor)

    with torch.no_grad():
        for i in tqdm(range(h_patches)):
            for j in range(w_patches):
                h_start = i * patch_size
                w_start = j * patch_size
                h_end = min(h_start + patch_size, height)
                w_end = min(w_start + patch_size, width)

                patch = input_tensor[:, :, h_start:h_end, w_start:w_end]

                # Pad the patch if it's smaller than patch_size
                pad_h = patch_size - (h_end - h_start)
                pad_w = patch_size - (w_end - w_start)
                if pad_h > 0 or pad_w > 0:
                    patch = torch.nn.functional.pad(patch, (0, pad_w, 0, pad_h))

                # Pass the patch through the model
                restored_patch = model(patch)

                # Remove padding from the restored patch
                restored_patch = restored_patch[:, :, :h_end-h_start, :w_end-w_start]

                restored_tensor[:, :, h_start:h_end, w_start:w_end] = restored_patch

    # De-normalize the tensors before calculating PSNR
    restored_tensor = restored_tensor.squeeze(0).cpu()
    restored_tensor = inv_transform(restored_tensor).clamp(0, 1)

    expected_tensor = expected_tensor.squeeze(0).cpu()
    expected_tensor = inv_transform(expected_tensor).clamp(0, 1)

    # Calculate PSNR
    psnr_value = calculate_psnr(expected_tensor, restored_tensor)
    # ssim_value = calculate_ssim(expected_tensor, restored_tensor)

    return restored_tensor, expected_tensor, psnr_value#, ssim_value

# Function to process a single image with seamless patching
def process_image_overlap(image_path, expected_path, model, transform, inv_transform, patch_size=128, overlap=2):
    image = Image.open(image_path).convert('RGB')
    expected_image = Image.open(expected_path).convert('RGB')
    
    input_tensor = transform(image).unsqueeze(0).to(device)
    expected_tensor = transform(expected_image).unsqueeze(0).to(device)

    _, _, height, width = input_tensor.shape

    # Calculate the number of patches needed with overlap
    stride = patch_size - overlap
    h_patches = (height + stride - 1) // stride
    w_patches = (width + stride - 1) // stride

    # Prepare tensors for output and weights for averaging
    restored_tensor = torch.zeros_like(input_tensor)
    weight_tensor = torch.zeros_like(input_tensor)

    with torch.no_grad():
        for i in tqdm(range(h_patches)):
            for j in range(w_patches):
                h_start = i * stride
                w_start = j * stride
                h_end = min(h_start + patch_size, height)
                w_end = min(w_start + patch_size, width)

                patch = input_tensor[:, :, h_start:h_end, w_start:w_end]

                # Pad the patch if it's smaller than patch_size
                pad_h = patch_size - (h_end - h_start)
                pad_w = patch_size - (w_end - w_start)
                if pad_h > 0 or pad_w > 0:
                    patch = torch.nn.functional.pad(patch, (0, pad_w, 0, pad_h))

                # Pass the patch through the model
                restored_patch = model(patch)

                # Remove padding from the restored patch
                restored_patch = restored_patch[:, :, :h_end-h_start, :w_end-w_start]

                # Add the restored patch to the restored tensor and update the weight tensor
                restored_tensor[:, :, h_start:h_end, w_start:w_end] += restored_patch
                weight_tensor[:, :, h_start:h_end, w_start:w_end] += 1

    # Average the restored tensor with the weight tensor
    restored_tensor /= weight_tensor

    # De-normalize the tensors before calculating PSNR
    restored_tensor = restored_tensor.squeeze(0).cpu()
    restored_tensor = inv_transform(restored_tensor).clamp(0, 1)

    expected_tensor = expected_tensor.squeeze(0).cpu()
    expected_tensor = inv_transform(expected_tensor).clamp(0, 1)

    # Calculate PSNR
    psnr_value = calculate_psnr(expected_tensor, restored_tensor)

    return restored_tensor, expected_tensor, psnr_value

def create_gaussian_kernel(size, sigma):
    """Create a Gaussian kernel."""
    ax = torch.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = torch.meshgrid([ax, ax])
    kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / torch.sum(kernel)

def process_image_smooth(image_path, expected_path, model, transform, inv_transform, patch_size=128, overlap=64):
    image = Image.open(image_path).convert('RGB')
    expected_image = Image.open(expected_path).convert('RGB')
    
    input_tensor = transform(image).unsqueeze(0).to(device)
    expected_tensor = transform(expected_image).unsqueeze(0).to(device)

    _, _, height, width = input_tensor.shape

    # Calculate the number of patches needed with overlap
    stride = patch_size - overlap
    h_patches = (height + stride - 1) // stride
    w_patches = (width + stride - 1) // stride

    # Prepare tensors for output and weights for averaging
    restored_tensor = torch.zeros_like(input_tensor)
    weight_tensor = torch.zeros_like(input_tensor)

    # Create a Gaussian kernel
    kernel_size = patch_size
    sigma = patch_size / 4  # You can adjust sigma for different smoothness levels
    gaussian_kernel = create_gaussian_kernel(kernel_size, sigma).to(device)

    with torch.no_grad():
        for i in tqdm(range(h_patches)):
            for j in range(w_patches):
                h_start = i * stride
                w_start = j * stride
                h_end = min(h_start + patch_size, height)
                w_end = min(w_start + patch_size, width)

                patch = input_tensor[:, :, h_start:h_end, w_start:w_end]

                # Pad the patch if it's smaller than patch_size
                pad_h = patch_size - (h_end - h_start)
                pad_w = patch_size - (w_end - w_start)
                if pad_h > 0 or pad_w > 0:
                    patch = torch.nn.functional.pad(patch, (0, pad_w, 0, pad_h))

                # Pass the patch through the model
                restored_patch = model(patch)

                # Apply Gaussian kernel
                actual_h = h_end - h_start
                actual_w = w_end - w_start
                gaussian_weight = gaussian_kernel[None, None, :actual_h, :actual_w]
                restored_patch = restored_patch[:, :, :actual_h, :actual_w] * gaussian_weight

                # Remove padding from the restored patch
                restored_patch = restored_patch[:, :, :h_end-h_start, :w_end-w_start]

                # Add the restored patch to the restored tensor and update the weight tensor
                restored_tensor[:, :, h_start:h_end, w_start:w_end] += restored_patch
                weight_tensor[:, :, h_start:h_end, w_start:w_end] += gaussian_weight

    # Average the restored tensor with the weight tensor
    restored_tensor /= weight_tensor

    # De-normalize the tensors before calculating PSNR
    restored_tensor = restored_tensor.squeeze(0).cpu()
    restored_tensor = inv_transform(restored_tensor).clamp(0, 1)

    expected_tensor = expected_tensor.squeeze(0).cpu()
    expected_tensor = inv_transform(expected_tensor).clamp(0, 1)

    # Calculate PSNR
    psnr_value = calculate_psnr(expected_tensor, restored_tensor)

    return restored_tensor, expected_tensor, psnr_value

def save_image(tensor, output_path):
    image = transforms.ToPILImage()(tensor)
    image.save(output_path)

# Main processing function
def process_images(input_folder, expected_folder, output_folder, checkpoint_path):
    os.makedirs(output_folder, exist_ok=True)
    model = load_model(checkpoint_path)
    transform = get_transform()
    inv_transform = get_inv_transform()

    psnr_values = 0

    for image_name in sorted(os.listdir(input_folder)):
        print(f"Processing {image_name}")
        image_path = os.path.join(input_folder, image_name)
        expected_path = os.path.join(expected_folder, image_name.replace('lq', 'hq'))
        
        restored_tensor, expected_tensor, psnr_value = process_image(image_path, expected_path, model, transform, inv_transform)

        # Save the restored image
        output_path = os.path.join(output_folder, image_name)
        save_image(restored_tensor, output_path)
        save_image(expected_tensor, output_path.replace('lq', 'hq'))
        
        # print(f"Saved {image_name}")
        print(f"Image: {image_name} PSNR: {psnr_value:.2f}")
        psnr_values += psnr_value
    avg = psnr_values/30

    print(f"Average PSNR: {avg:.2f}")#, SSIM: {ssim_value:.4f}")

    print(f"Restored images saved to {output_folder}")

if __name__ == '__main__':
    input_folder = '../../../test_pol_cre/LQ'
    expected_folder  = '../../../test_pol_cre/HQ'
    output_folder = '../final_pol_test_seamless'
    # checkpoint_path = '../org_chkpt/checkpoint_msr_36.pt'
    checkpoint_path = '../poled_chkpt/checkpoint_pol_msr_30.pt'
    # checkpoint_path = '../best_msr.pt'
    process_images(input_folder, expected_folder, output_folder, checkpoint_path)