import torch
from torch.utils.data import Dataset
import numpy as np


class MAXIM_Dataset_multiplechr_tri_patches(Dataset):
    def __init__(self, low_data_dir, high_data_dir, chr):
        super(MAXIM_Dataset_multiplechr_tri_patches, self).__init__()
        self.low_resolution = np.load(low_data_dir + str(chr[0]) + '_Type3_patches.npz', allow_pickle=True)["matrix"]
        self.high_resolution = np.load(high_data_dir + str(chr[0]) + '_Type3_patches.npz', allow_pickle=True)["matrix"]
        for i in range(1, len(chr)):
            low = np.load(low_data_dir + str(chr[i]) + '_Type3_patches.npz', allow_pickle=True)["matrix"]
            high = np.load(high_data_dir + str(chr[i]) + '_Type3_patches.npz', allow_pickle=True)["matrix"]
            self.low_resolution = np.vstack((self.low_resolution, low))
            self.high_resolution = np.vstack((self.high_resolution, high))
            print(chr[i], 'done')


    def __len__(self):
        return self.low_resolution.shape[0]

    def __getitem__(self, idx):
        low_matrix = torch.FloatTensor(self.low_resolution[idx]).unsqueeze(0)
        high_matrix = torch.FloatTensor(self.high_resolution[idx]).unsqueeze(0)
        return {'low': low_matrix, 'high': high_matrix}





