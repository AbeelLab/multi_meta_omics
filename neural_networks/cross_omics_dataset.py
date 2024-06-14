import torch

from torch.utils.data import Dataset

class CrossOmicsDataset(Dataset):
    def __init__(self, in_data, out_data):
        self.in_data = torch.from_numpy(in_data).to("cpu").to(torch.float)
        self.out_data = torch.from_numpy(out_data).to("cpu").to(torch.float)

    def __len__(self):
        return len(self.in_data)

    def __getitem__(self, idx):
        return self.in_data[idx], self.out_data[idx]
