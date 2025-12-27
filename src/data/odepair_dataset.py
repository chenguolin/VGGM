from typing import *

import os
import torch
from torch.utils.data import Dataset

from src.options import ROOT


class OdepairDataset(Dataset):
    def __init__(self, root: str):
        super().__init__()

        files = os.listdir(root)
        self.file_paths = [
            os.path.join(root, file)
            for file in files if file.endswith(".pt")
        ]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        ode_pair = torch.load(file_path, weights_only=False)  # 'noisy_latents', 'prompt_embeds', 'plucker_embeds', (optional) 'cond_latents'
        return ode_pair
        # {
        #     "noisy_latents"   # (T+1, C, f, h, w)
        #     "prompt_embeds"   # (N, D')
        #     "plucker_embeds"  # (D, f, hh, ww)
        #     "cond_latents"    # (C, 1, h, w); only for I2V
        # }
