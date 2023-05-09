import lightning
import torch
import torch.nn as nn
from gpt_arch import GPT2
from train import GPTTrainer
import numpy as np
import torchvision
import torch.utils.data as data
from utils import dequantize, quantize_image
import torch.nn.functional as F
from matplotlib import pyplot as plt
import os

if __name__ == "__main__":

    ckpt_path = "./model_ckpt/lightning_logs/version_3/checkpoints/epoch=31-step=10000.ckpt"
    # checkpoint = torch.load(ckpt_path)

    centroids = np.load("./cifar10_data/cifar10_centroids_512.npy")
    centroids = torch.tensor(centroids)

    model = GPTTrainer.load_from_checkpoint(ckpt_path).model
    model.eval()

    train_dataset = torchvision.datasets.CIFAR10(root="./cifar10_data", train=True, download=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))
    # use 20% of training data for validation
    train_set_size = int(len(train_dataset) * 0.8)
    valid_set_size = len(train_dataset) - train_set_size
    # split the train set into two
    seed = torch.Generator().manual_seed(42)
    train_set, valid_set = data.random_split(train_dataset, [train_set_size, valid_set_size], generator=seed)
    val_dataloader = iter(torch.utils.data.DataLoader(valid_set, batch_size=1))

    num_sample_images = 5
    context_frac = 0.5
    os.makedirs("./generated_images", exist_ok=True)
    disp_imgs = []
    for gen_idx in range(num_sample_images): 

        sample = next(val_dataloader)[0] # [B,c,h,w]
        b,c,h,w = list(sample.shape)
        disp_imgs.append(sample)

        # org_image = sample.permute(0,2,3,1).contiguous().reshape(h,w,c).numpy()

        seq_len = h*w
        sample = quantize_image(sample, centroids) # [B,h,w]
        sample = sample.reshape(-1, seq_len) # [B, seq_len]
        sample = sample.transpose(0,1).contiguous() # [seq_len, B]

        context_len = int(context_frac*seq_len)
        rem_len = seq_len - context_len

        context = sample[0:context_len, :] # [seq_len/2, B]
        context_img = dequantize(torch.cat((context, torch.zeros((rem_len, b), dtype=torch.long)), dim=0).squeeze(), centroids)
        disp_imgs.append(context_img.reshape(b,h,w,c).permute(0,3,1,2).contiguous())
        # rem_len = seq_len - seq_len//2

        pad_seq = torch.zeros(1, b, dtype=torch.long)
        for idx in range(rem_len):
            print(idx)
            model_input = torch.cat((context, pad_seq), dim=0) # [seq_len/2 + 1, B]
            logits = model(model_input) # [seq_len/2 + 1, B, embed_dim]
            
            logits = logits[-1, :, :] # [B, embed_dim]
            probs = F.softmax(logits, dim=-1) # [B, embed_dim]
            preds = torch.argmax(probs, dim=-1, keepdim=True).transpose(1,0) # [1, B]
            context = torch.cat((context, preds), dim=0) # [seq_len/2 + 1, B] 

        deq_img = dequantize(context.squeeze(), centroids)
        final_image = deq_img.reshape(b,h,w,c).permute(0,3,1,2).contiguous()
        disp_imgs.append(final_image)

    disp_imgs = torch.cat(disp_imgs)

    torchvision.utils.save_image(torchvision.utils.make_grid(disp_imgs, nrow=3), fp=f"./gen_img_{num_sample_images}_{context_frac}.png")
