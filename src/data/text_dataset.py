from typing import *

from torch.utils.data import Dataset

from src.options import ROOT


class TextDataset(Dataset):
    def __init__(self,
        prompt_path: str = f"{ROOT}/.cache/vidprom_filtered_extended.txt",
    ):
        with open(prompt_path, encoding="utf-8") as f:
            self.prompt_list = [line.rstrip() for line in f]

    def __len__(self):
        return len(self.prompt_list)

    def __getitem__(self, idx):
        batch = {
            "uid": idx,
            "prompt": self.prompt_list[idx],
        }
        return batch
