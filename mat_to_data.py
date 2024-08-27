import os
import numpy as np
import scipy.io
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

from tqdm import tqdm 

class MatDataset(Dataset):
    def __init__(self, input_mat_file_path=r'UDC/Test/poled_test_display.mat', gt_mat_file_path=r'UDC/Test/poled_test_gt.mat', input_key='test_display', gt_key='test_gt', target_size=(128, 128)):
        self.input_mat_file_path = input_mat_file_path
        self.gt_mat_file_path = gt_mat_file_path
        self.input_key = input_key
        self.gt_key = gt_key
        self.target_size = target_size
        self.inputs, self.gts = self._load_mat_files()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])

    def _load_mat_files(self):
        # Load data from .mat files
        input_data = scipy.io.loadmat(self.input_mat_file_path)[self.input_key]
        gt_data = scipy.io.loadmat(self.gt_mat_file_path)[self.gt_key]

        inputs = []
        gts = []

        for i in range(input_data.shape[0]):
            try:
                img = input_data[i]
                inputs.append(img)
            except Exception as e:
                print(f"Skipping input image {i} due to error: {e}")
                continue

        for i in range(gt_data.shape[0]):
            try:
                img = gt_data[i]
                gts.append(img)
            except Exception as e:
                print(f"Skipping gt image {i} due to error: {e}")
                continue

        return inputs, gts

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_img = self.inputs[idx]
        gt_img = self.gts[idx]
        input_img = self.transform(input_img)
        gt_img = self.transform(gt_img)
        return input_img, gt_img

    def save_images_to_folders(self, lq_folder='test_pol/LQ', hq_folder='test_pol/HQ'):
        os.makedirs(lq_folder, exist_ok=True)
        os.makedirs(hq_folder, exist_ok=True)

        for i in tqdm(range(len(self.inputs))):
            input_img = Image.fromarray(self.inputs[i])
            gt_img = Image.fromarray(self.gts[i])

            input_img.save(os.path.join(lq_folder, f'lq_{i}.png'))
            gt_img.save(os.path.join(hq_folder, f'hq_{i}.png'))


if __name__ == '__main__':
    #=======================poled test=======================
    dataset1 = MatDataset(
        input_mat_file_path=r'dataset/UDC/Test/poled_test_display.mat', 
        gt_mat_file_path=r'dataset/UDC/Test/poled_test_gt.mat'
    )
    dataset1.save_images_to_folders(lq_folder='dataset/UDC/Test/Poled/LQ', hq_folder='dataset/UDC/Test/Poled/HQ')
    #=======================toled test=======================
    dataset2 = MatDataset(
        input_mat_file_path=r'dataset/UDC/Test/toled_test_display.mat', 
        gt_mat_file_path=r'dataset/UDC/Test/toled_test_gt.mat'
    )
    dataset2.save_images_to_folders(lq_folder='dataset/UDC/Test/Toled/LQ', hq_folder='dataset/UDC/Test/Toled/HQ')
    #=======================poled val=======================
    dataset3 = MatDataset(
        input_mat_file_path=r'dataset/UDC/Val/poled_val_display.mat', 
        gt_mat_file_path=r'dataset/UDC/Val/poled_test_gt.mat'
    )
    dataset3.save_images_to_folders(lq_folder='dataset/UDC/Val/Poled/LQ', hq_folder='dataset/UDC/Val/Poled/HQ')
    #=======================toled val=======================
    dataset4 = MatDataset(
        input_mat_file_path=r'dataset/UDC/Val/toled_val_display.mat', 
        gt_mat_file_path=r'dataset/UDC/Val/toled_val_gt.mat'
    )
    dataset4.save_images_to_folders(lq_folder='dataset/UDC/Val/Toled/LQ', hq_folder='dataset/UDC/Val/Toled/HQ')