import torch
import wandb  # Added for Weights & Biases integration
torch.cuda.empty_cache()

import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tqdm import tqdm
import os
import torchvision.transforms as transforms

from custom_dataset import TrainAllDataset, VallAll
from mambair.mambairunet_arch import MambaIRUNet
from mambair.testing.tester import process_image, get_transform, get_inv_transform
#====================================Log in to Weights & Biases===============================
wandb.login()

lr = 0.00025
batch_size = 4
num_epochs = 100
patch_size = 64

wandb.init(project='mambairunet-training', config={
    "learning_rate": lr,
    "batch_size": batch_size,
    "num_epochs": num_epochs
})
config = wandb.config

#====================================model loading===============================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model = MambaIRUNet()
model = model.cuda()

checkpoint_path = 'toled_chkpt/checkpoint_tol_msr_12.pt'  # Path to the last checkpoint
# checkpoint_path = None
if checkpoint_path and os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    print(f'Loaded checkpoint from epoch {start_epoch}')
else:
    start_epoch = 0

optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = torch.nn.L1Loss()
train_dataset = TrainAllDataset(patch_size=patch_size)
val_dataset = VallAll(patch_size=patch_size)

scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

#====================================model training===============================
best_val_loss = float('inf')
val_transform = get_transform()
val_inv_transform = get_inv_transform()

for epoch in range(start_epoch, num_epochs+start_epoch):
    model.train()
    epoch_loss = 0.0
    loss_file = open("epoch_losses.txt", "w")

    for (hq_imgs, lq_imgs) in tqdm(train_dataloader):

        hq_imgs = hq_imgs.permute(1, 0, 2, 3, 4)
        lq_imgs = lq_imgs.permute(1, 0, 2, 3, 4)

        for hq_img, lq_img in zip(hq_imgs,lq_imgs):
            hq_img, lq_img = hq_img.cuda(), lq_img.cuda()
            optimizer.zero_grad()
            output = model(hq_img)
            loss = criterion(output, lq_img)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

    avg_epoch_loss = epoch_loss / len(train_dataloader)
    print(f'Training Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}')
    wandb.log({"avg_train_loss": avg_epoch_loss, "epoch": epoch + 1})

    # Validation loop
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for (hq_imgs, lq_imgs) in tqdm(val_dataloader):
            
            hq_imgs = hq_imgs.permute(1, 0, 2, 3, 4)
            lq_imgs = lq_imgs.permute(1, 0, 2, 3, 4)

            for hq_img, lq_img in zip(hq_imgs,lq_imgs):
                hq_img, lq_img = hq_img.cuda(), lq_img.cuda()
                output = model(hq_img)
                loss = criterion(output, lq_img)
                val_loss += loss.item()

    avg_val_loss = val_loss / len(val_dataloader)
    print(f'Validation Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_val_loss:.4f}')
    wandb.log({"avg_val_loss": avg_val_loss})

    # Record losses
    loss_file.write(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_epoch_loss:.4f}, Validation Loss: {avg_val_loss:.4f}\n')

    #example
    sample_images = []

    image_path1 = '../../test_tol_cre/LQ/lq_8.png'
    gt_path1 = '../../test_tol_cre/HQ/hq_8.png'
    restored1, gt1, psnr1 = process_image(image_path1, gt_path1, model, val_transform, val_inv_transform, patch_size=128)
    sample_images.append(wandb.Image(restored1, caption=f'Output Image1 {psnr1}'))

    image_path2 = '../../test_tol_cre/LQ/lq_17.png'
    gt_path2 = '../../test_tol_cre/HQ/hq_17.png'
    restored2, gt2, psnr2 = process_image(image_path2, gt_path2, model, val_transform, val_inv_transform, patch_size=128)
    sample_images.append(wandb.Image(restored2, caption=f'Output Image2 {psnr2}'))

    wandb.log({"examples": sample_images})

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'best_msr.pt')

    # Save checkpoint every epoch
    if (epoch + 1) % 1 == 0:
        checkpoint_path = f'toled_chkpt/checkpoint_tol_msr_{epoch + 1}.pt'
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': avg_epoch_loss,
            'val_loss': avg_val_loss
        }, checkpoint_path)

    # Adjust learning rate based on validation loss
    scheduler.step(avg_val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    print(f'Adjusted Learning Rate: {current_lr}')
    wandb.log({"adjusted_learning_rate": current_lr, "epoch": epoch + 1})
    loss_file.close()

print('Finished Training and Validation')