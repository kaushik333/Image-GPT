import torch
import torch.nn as nn
from gpt_arch import GPT2
import numpy as np
import torchvision
from utils import quantize_image
import lightning as pl
import torch.utils.data as data
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torch.optim.lr_scheduler import LambdaLR
import math

class GPTTrainer(pl.LightningModule):
    def __init__(self, model, centroids, lr, warmup_steps, steps):
        super().__init__()
        assert torch.is_tensor(centroids)

        self.centroids = centroids
        self.model = model
        self.learning_rate = lr

        self.warmup_steps = warmup_steps
        self.total_steps = steps

        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        data, _ = batch

        x = quantize_image(data, self.centroids.to(data)) # [B, H, W]
        x = x.view(x.shape[0], -1) # [B, seq_len]; seq_len=HxW
        x = x.transpose(0,1).contiguous() # [seq_len, B]

        logits = self.model(x) # [seq_len, B, embed_dim]
        train_loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), x.view(-1))
        self.log("train_loss", train_loss)

        return {"train_loss": train_loss}

    def validation_step(self, batch, batch_idx):
        data, _ = batch

        x = quantize_image(data, self.centroids.to(data)) # [B, H, W]
        x = x.view(x.shape[0], -1) # [B, seq_len]; seq_len=HxW
        x = x.transpose(0,1).contiguous() # [seq_len, B]

        logits = self.model(x) # [seq_len, B, embed_dim]
        val_loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), x.view(-1))

        self.log("val_loss", val_loss)

        return {"val_loss": val_loss}

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, betas=(0.9, 0.95))

        scheduler = {
            "scheduler": LambdaLR(
                optimizer, self.learning_rate_fn
            ),
            "interval": "step",
        }
        return [optimizer], [scheduler]

    def learning_rate_fn(self, step):

        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        else:
            progress = float(step - self.warmup_steps) / float(
                max(1, self.total_steps - self.warmup_steps)
            )
            return 0.5 * (1.0 + math.cos(math.pi * progress))

if __name__ == "__main__":

    ######################
    # Define input params
    ######################

    # XXSmall-GPT
    # num_clusters = 16
    # centroids_path = f"./cifar10_data/cifar10_centroids_{num_clusters}.npy"
    # cifar_data_path = "./cifar10_data"
    # img_h = 32
    # img_w = 32
    # transformer_num_blocks = 8
    # transformer_num_heads = 2
    # warmup_steps = 500
    # total_steps = 25000
    # embed_dim = 16
    # model_save_dir = "./model_ckpt"
    # batch_size=64
    # accumulate_grad_batches=1

    # Small-GPT
    num_clusters = 512
    centroids_path = f"./cifar10_data/cifar10_centroids_{num_clusters}.npy"
    cifar_data_path = "./cifar10_data"
    img_h = 32
    img_w = 32
    transformer_num_blocks = 24
    transformer_num_heads = 8
    warmup_steps = 500
    total_steps = 10_000
    embed_dim = 512
    model_save_dir = "./model_ckpt"
    batch_size=16
    accumulate_grad_batches=4

    centroids = nn.Parameter(torch.from_numpy(np.load(centroids_path)), requires_grad=False)

    gpt2 = GPT2(embed_dim=embed_dim, num_vocab=centroids.shape[0], img_h=img_h, img_w=img_w, num_blocks=transformer_num_blocks, num_heads=transformer_num_heads) #24, 8
    GPTModel = GPTTrainer(gpt2, centroids, lr=1e-4, warmup_steps=warmup_steps, steps=total_steps)
    train_dataset = torchvision.datasets.CIFAR10(root=cifar_data_path, train=True, download=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))
    
    # use 20% of training data for validation
    train_set_size = int(len(train_dataset) * 0.8)
    valid_set_size = len(train_dataset) - train_set_size

    # split the train set into two
    seed = torch.Generator().manual_seed(42)
    train_set, valid_set = data.random_split(train_dataset, [train_set_size, valid_set_size], generator=seed)

    # find optimal learning_rate and set it
    # lr_train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
    # lrfind_trainer = pl.Trainer(accelerator="gpu", devices=[0], auto_lr_find=True, default_root_dir=model_save_dir)
    # lr_finder = lrfind_trainer.tuner.lr_find(GPTModel, lr_train_dataloader)
    # GPTModel.hparams.learning_rate = lr_finder.suggestion()

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
    val_dataloader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size)

    # log only 3 times every epoch to save time.
    log_every_n_steps = (len(train_dataloader) + len(val_dataloader))//2
    trainer = pl.Trainer(max_steps=total_steps, accelerator="gpu", devices=[2, 3], default_root_dir="./model_ckpt", log_every_n_steps=log_every_n_steps, accumulate_grad_batches=accumulate_grad_batches)
    trainer.fit(model=GPTModel, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)