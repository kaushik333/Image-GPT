import torch
import numpy as np

def quantize_image(image_batch, centroids):
    
    assert torch.is_tensor(image_batch)
    assert torch.is_tensor(centroids)
    assert len(image_batch.shape) == 4 # [B,C,H,W]
    assert len(centroids.shape) == 2

    b,c,h,w = image_batch.shape
    image_batch = image_batch.permute(0,2,3,1).contiguous() # [B,H,W,C]
    image_batch = image_batch.view(-1,c)
    assert image_batch.shape == (b*h*w, c)

    image_batch = torch.unsqueeze(image_batch, 0) # [1,(BxHxW),C]
    centroids = torch.unsqueeze(centroids, 0) # [1,num_centroids,C]

    euc_dists = torch.cdist(image_batch, centroids).squeeze(0) # [(BxHxW),num_centroids]
    assert euc_dists.shape == (b*h*w, centroids.shape[1])

    quantized_image = torch.argmin(euc_dists, 1) # [(BxHxW)]
    quantized_reshaped = quantized_image.view(b,h,w)

    return quantized_reshaped

def dequantize(quantized_flattened_image, centroids):

    assert torch.is_tensor(quantized_flattened_image)
    assert torch.is_tensor(centroids)
    assert len(quantized_flattened_image.shape) == 1
    assert len(centroids.shape) == 2

    dequantized_image = centroids[quantized_flattened_image] # [(BxHxW)x3]

    return dequantized_image