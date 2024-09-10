import numpy as np

import torch
import torch.nn.functional as F

class Dataset:
    def __init__(self, data_table, device="cpu", dtype=torch.float32, seed=0):
        self.data_table = data_table
        self.device = device
        self.dtype = dtype
        # Set seed
        self.seed = seed

    def __getitem__(self, index):
        sample = self.data_table[index]
        input_ids = torch.tensor(sample["input_ids"], dtype=torch.long, device=self.device)
        values = torch.tensor(sample["values"], dtype=self.dtype, device=self.device)
        labels = torch.tensor(sample["labels"], dtype=torch.long, device=self.device)
        return {
            "input_ids": input_ids,
            "values": values,
            "labels": labels,
        }

    def __len__(self):
        return int(self.data_table)