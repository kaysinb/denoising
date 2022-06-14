import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import RandomCrop

from src.settings import *


def get_mel_df(path: Path):
    sounds_clean = []
    sounds_noisy = []
    path = path / 'clean'
    for folder in path.iterdir():
        for file in folder.iterdir():
            sounds_clean.append(np.load(file))
            sounds_noisy.append(np.load(str(file).replace('clean', 'noisy')))

    mel_df = pd.DataFrame({'sounds_clean': sounds_clean, 'sounds_noisy': sounds_noisy})
    return mel_df


class MelDataset(Dataset):
    # Data Loading
    def __init__(self, mel_df):
        self.sounds_clean = mel_df['sounds_clean'].values
        self.sounds_noisy = mel_df['sounds_noisy'].values
        self.n_samples = len(self.sounds_clean)
        self.random_crop = RandomCrop((M_SIZE, T_SIZE))

    # Indexing dataset[index]
    def __getitem__(self, index):
        sample_clean = self.sounds_clean[index].T
        sample_noisy = self.sounds_noisy[index].T

        if sample_clean.shape[1] > T_SIZE:
            sample = np.append(sample_clean[None, :, :], sample_noisy[None, :, :],
                               axis=0)
            sample = torch.tensor(sample).type(torch.FloatTensor)
            sample = self.random_crop(sample)
            sample_clean = sample[0]
            sample_noisy = sample[1]
        else:
            sample_clean = extend_sample(sample_clean)
            sample_clean = torch.tensor(sample_clean).type(torch.FloatTensor)
            sample_noisy = extend_sample(sample_noisy)
            sample_noisy = torch.tensor(sample_noisy).type(torch.FloatTensor)
        return sample_clean[None, :, :], sample_noisy[None, :, :]

    # len(dataset)
    def __len__(self):
        return self.n_samples


def extend_sample(sample):
    if sample.shape[1] > 0:
        while sample.shape[1] < T_SIZE:
            sample = np.hstack((sample, sample))
    return sample[:, :T_SIZE]


def get_dataloader(path: Path):
    mel_df = get_mel_df(path)
    dataset = MelDataset(mel_df)

    train_part = int(len(dataset) * 5 / 6)
    val_part = int(len(dataset) * 1 / 6)

    train_set, val_set = torch.utils.data.random_split(dataset, [train_part, val_part],
                                                       generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_set, batch_size=BATCH_SIZE, shuffle=True)

    return train_loader, val_loader
