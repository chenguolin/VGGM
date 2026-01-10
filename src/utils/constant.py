import torch


ZERO_VAE_CACHE = [
    torch.zeros(1, 16, 2, 36, 64),
    torch.zeros(1, 384, 2, 36, 64),
    torch.zeros(1, 384, 2, 36, 64),
    torch.zeros(1, 384, 2, 36, 64),
    torch.zeros(1, 384, 2, 36, 64),
    torch.zeros(1, 384, 2, 36, 64),
    torch.zeros(1, 384, 2, 36, 64),
    torch.zeros(1, 384, 2, 36, 64),
    torch.zeros(1, 384, 2, 36, 64),
    torch.zeros(1, 384, 2, 36, 64),
    torch.zeros(1, 384, 2, 36, 64),
    torch.zeros(1, 384, 2, 36, 64),
    torch.zeros(1, 192, 2, 72, 128),
    torch.zeros(1, 384, 2, 72, 128),
    torch.zeros(1, 384, 2, 72, 128),
    torch.zeros(1, 384, 2, 72, 128),
    torch.zeros(1, 384, 2, 72, 128),
    torch.zeros(1, 384, 2, 72, 128),
    torch.zeros(1, 384, 2, 72, 128),
    torch.zeros(1, 192, 2, 144, 256),
    torch.zeros(1, 192, 2, 144, 256),
    torch.zeros(1, 192, 2, 144, 256),
    torch.zeros(1, 192, 2, 144, 256),
    torch.zeros(1, 192, 2, 144, 256),
    torch.zeros(1, 192, 2, 144, 256),
    torch.zeros(1, 96, 2, 288, 512),
    torch.zeros(1, 96, 2, 288, 512),
    torch.zeros(1, 96, 2, 288, 512),
    torch.zeros(1, 96, 2, 288, 512),
    torch.zeros(1, 96, 2, 288, 512),
    torch.zeros(1, 96, 2, 288, 512),
    torch.zeros(1, 96, 2, 288, 512)
]

feat_names = [f"vae_cache_{i}" for i in range(len(ZERO_VAE_CACHE))]
ALL_INPUTS_NAMES = ["z", "use_cache"] + feat_names
