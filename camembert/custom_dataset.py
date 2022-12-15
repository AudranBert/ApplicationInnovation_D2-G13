import pandas as pd
import torch
from torch.utils.data import Dataset
import to_csv as dm
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)

class CustomDataset(Dataset):

    def __init__(self, dataset, note=True):
        self.note = note
        df = dm.dataset_to_pickle(dataset, note)
        self.x = df['commentaire'].to_numpy()
        if self.note:
            self.y = df['note'].to_numpy()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if self.note:
            # logging.info(self.y[idx])
            x = torch.Tensor(self.x[idx])
            return x.to(torch.int), torch.Tensor(np.asarray(self.y[idx]))
        return torch.Tensor.int(self.x[idx])

