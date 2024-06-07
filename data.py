import math
import numpy as np
import torch


class Dataset(torch.utils.data.Dataset):

    def __init__(self, x_set, y_set, batch_size, shuffle=False):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.shuffle=shuffle

    def __len__(self):
        return len(self.x)

		# batch 단위로 직접 묶어줘야 함
    def __getitem__(self, idx):

        return np.squeeze(self.x[idx]), np.squeeze(self.y[idx])