import torch
import wandb  # Added for Weights & Biases integration
torch.cuda.empty_cache()

import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tqdm import tqdm
import os
import torchvision.transforms as transforms
import argparse

from custom_dataset import AllDataset
# from resmambair.resmambairunet_arch import MambaIRUNet
from resmambair.testing.tester import process_image, get_transform, get_inv_transform

def main(args):
    #====================================Log in to Weights & Biases===============================
    wandb.login()

    wandb.init(project='mambairunet-training', config={
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs
    })
    config = wandb.config

    #====================================model loading===============================
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    model = MambaIRUNet().cuda()

    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f'Loaded checkpoint from epoch {start_epoch}')
    else:
        start_epoch = 0

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.L1Loss()

    #====================================data loading===============================
    train_dataset = AllDataset(patch_size=args.patch_size, path = r'dataset/UDC/Train/{}'.format(args.dataset))
    val_dataset = AllDataset(patch_size=args.patch_size, path = r'dataset/UDC/Val/Poled/{}'.format(args.dataset))

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    #====================================model training===============================
    best_val_loss = float('inf')
    val_transform = get_transform()
    val_inv_transform = get_inv_transform()

    for epoch in range(start_epoch, args.num_epochs + start_epoch):
        model.train()
        epoch_loss = 0.0
        with open("epoch_losses.txt", "w") as loss_file:

            for (hq_imgs, lq_imgs) in tqdm(train_dataloader):
                hq_imgs = hq_imgs.permute(1, 0, 2, 3, 4)
                lq_imgs = lq_imgs.permute(1, 0, 2, 3, 4)

                for hq_img, lq_img in zip(hq_imgs, lq_imgs):
                    hq_img, lq_img = hq_img.cuda(), lq_img.cuda()
                    optimizer.zero_grad()
                    output = model(hq_img)
                    loss = criterion(output, lq_img)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / len(train_dataloader)
            print(f'Training Epoch [{epoch + 1}/{args.num_epochs}], Loss: {avg_epoch_loss:.4f}')
            wandb.log({"avg_train_loss": avg_epoch_loss, "epoch": epoch + 1})

            # Validation loop
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for (hq_imgs, lq_imgs) in tqdm(val_dataloader):
                    hq_imgs = hq_imgs.permute(1, 0, 2, 3, 4)
                    lq_imgs = lq_imgs.permute(1, 0, 2, 3, 4)

                    for hq_img, lq_img in zip(hq_imgs, lq_imgs):
                        hq_img, lq_img = hq_img.cuda(), lq_img.cuda()
                        output = model(hq_img)
                        loss = criterion(output, lq_img)
                        val_loss += loss.item()

            avg_val_loss = val_loss / len(val_dataloader)
            print(f'Validation Epoch [{epoch + 1}/{args.num_epochs}], Loss: {avg_val_loss:.4f}')
            wandb.log({"avg_val_loss": avg_val_loss})

            # Record losses
            loss_file.write(f'Epoch [{epoch + 1}/{args.num_epochs}], Training Loss: {avg_epoch_loss:.4f}, Validation Loss: {avg_val_loss:.4f}\n')

            # Example images
            try:
                sample_images = []
                image_paths = [
                    (r'dataset/UDC/Test/{}/LQ/lq_17.png'.format(args.dataset), r'dataset/UDC/Test/{}/HQ/hq_17.png'.format(args.dataset)),
                    (r'dataset/UDC/Test/{}/LQ/lq_8.png'.format(args.dataset), r'dataset/UDC/Test/{}/HQ/hq_8.png'.format(args.dataset)),
                ]

                for lq_path, hq_path in image_paths:
                    restored, gt, psnr = process_image(lq_path, hq_path, model, val_transform, val_inv_transform, patch_size=128)
                    sample_images.append(wandb.Image(restored, caption=f'Output Image PSNR: {psnr:.2f}'))

                wandb.log({"examples": sample_images})
            except:
                print("Error processing example images")

            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), 'best_msr.pt')

            # Save checkpoint every epoch
            checkpoint_path = f'chkpt/resmamba_{args.dataset}_{epoch + 1}.pt'
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

    print('Finished Training and Validation')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MambaIRUNet Training Script')

    parser.add_argument('--dataset', type=str, default='Poled', help='Poled or Toled (data to be trained)')
    parser.add_argument('--lr', type=float, default=0.00025, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--patch_size', type=int, default=64, help='Patch size')
    parser.add_argument('--cuda_device', type=str, default="0", help='CUDA device number')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to the last checkpoint')

    args = parser.parse_args()

    main(args)
