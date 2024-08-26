import os
import cv2
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import scipy.io
import numpy as np 
from natsort import natsorted 

class TrainAllDataset(Dataset):
    def __init__(self, patch_size=256):

        self.patch_size = patch_size
        self.poled_path = r'/home/santhi/UDC/UDC/Train/Poled'
        self.toled_path = r'/home/santhi/UDC/UDC/Train/Toled'

        self.poled_hq_path = os.path.join(self.poled_path, 'HQ')
        self.poled_lq_path = os.path.join(self.poled_path, 'LQ')

        self.toled_hq_path = os.path.join(self.toled_path, 'HQ')
        self.toled_lq_path = os.path.join(self.toled_path, 'LQ')

        self.hq_img_paths = []
        self.lq_img_paths = []

        self.hq_img_paths.extend([os.path.join(self.toled_hq_path, img) for img in os.listdir(self.toled_hq_path)])
        self.lq_img_paths.extend([os.path.join(self.toled_lq_path, img) for img in os.listdir(self.toled_lq_path)])
        # self.hq_img_paths.extend([os.path.join(self.poled_hq_path, img) for img in os.listdir(self.poled_hq_path)])
        # self.lq_img_paths.extend([os.path.join(self.poled_lq_path, img) for img in os.listdir(self.poled_lq_path)])

        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        # Transformation
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.hq_img_paths)
        # return 2

    def __getitem__(self, idx):

        while True: 
            hq_path = self.hq_img_paths[idx]
            lq_path = self.lq_img_paths[idx]

            hq_img = cv2.imread(hq_path)
            lq_img = cv2.imread(lq_path)

            if hq_img is None:
                idx = (idx + 1) % len(self.poled_hq_img_path)
                continue

            if lq_img is None:
                idx = (idx + 1) % len(self.poled_lq_img_path)
                continue

            hq_img = cv2.cvtColor(hq_img, cv2.COLOR_BGR2RGB)
            lq_img = cv2.cvtColor(lq_img, cv2.COLOR_BGR2RGB)

            hq_img_pil = Image.fromarray(hq_img)
            lq_img_pil = Image.fromarray(lq_img)

            hq_img_resized = self.transform(hq_img_pil)
            lq_img_resized = self.transform(lq_img_pil)

            lq_img_resized, hq_img_resized = self.tiles(lq_img_resized, hq_img_resized, self.patch_size)

            return lq_img_resized , hq_img_resized
        
    def tiles(self, img, mask, patch_size):

        img_patches = img.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
        img_patches  = img_patches.contiguous().view(3,-1, patch_size, patch_size) 
        img_patches = img_patches.permute(1,0,2,3)
        
        mask_patches = mask.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
        mask_patches = mask_patches.contiguous().view(3,-1, patch_size, patch_size)
        mask_patches = mask_patches.permute(1,0,2,3)
        
        return img_patches, mask_patches



class VallAll(Dataset):
    def __init__(self, patch_size=256):

        self.patch_size = patch_size
        self.poled_path = r'/home/santhi/UDC/VAL_pol_cre'
        self.toled_path = r'/home/santhi/UDC/VAL_tol_cre'

        self.poled_hq_path = os.path.join(self.poled_path, 'HQ')
        self.poled_lq_path = os.path.join(self.poled_path, 'LQ')

        self.toled_hq_path = os.path.join(self.toled_path, 'HQ')
        self.toled_lq_path = os.path.join(self.toled_path, 'LQ')

        self.hq_img_paths = []
        self.lq_img_paths = []

        self.hq_img_paths.extend([os.path.join(self.toled_hq_path, img) for img in os.listdir(self.toled_hq_path)])
        self.lq_img_paths.extend([os.path.join(self.toled_lq_path, img) for img in os.listdir(self.toled_lq_path)])
        # self.hq_img_paths.extend([os.path.join(self.poled_hq_path, img) for img in os.listdir(self.poled_hq_path)])
        # self.lq_img_paths.extend([os.path.join(self.poled_lq_path, img) for img in os.listdir(self.poled_lq_path)])


        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        # Transformation
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.hq_img_paths)
        # return 2

    def __getitem__(self, idx):

        while True: 
            hq_path = self.hq_img_paths[idx]
            lq_path = self.lq_img_paths[idx]

            hq_img = cv2.imread(hq_path)
            lq_img = cv2.imread(lq_path)

            if hq_img is None:
                idx = (idx + 1) % len(self.poled_hq_img_path)
                continue

            if lq_img is None:
                idx = (idx + 1) % len(self.poled_lq_img_path)
                continue


            hq_img = cv2.cvtColor(hq_img, cv2.COLOR_BGR2RGB)
            lq_img = cv2.cvtColor(lq_img, cv2.COLOR_BGR2RGB)

            hq_img_pil = Image.fromarray(hq_img)
            lq_img_pil = Image.fromarray(lq_img)

            hq_img_resized = self.transform(hq_img_pil)
            lq_img_resized = self.transform(lq_img_pil)


            lq_img_resized, hq_img_resized = self.tiles(lq_img_resized, hq_img_resized, self.patch_size)

            return lq_img_resized ,  hq_img_resized
        
    
    def tiles(self, img, mask, patch_size):

        img_patches = img.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
        img_patches  = img_patches.contiguous().view(3,-1, patch_size, patch_size)
        img_patches = img_patches.permute(1,0,2,3)
        
        mask_patches = mask.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
        mask_patches = mask_patches.contiguous().view(3,-1, patch_size, patch_size)
        mask_patches = mask_patches.permute(1,0,2,3)
        
        return img_patches, mask_patches
