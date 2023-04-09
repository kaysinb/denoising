import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import pydoc
from src.settings import *


def object_from_dict(d, parent=None, **default_kwargs):
    kwargs = d.copy()
    object_type = kwargs.pop("type")
    for name, value in default_kwargs.items():
        kwargs.setdefault(name, value)
    if parent is not None:
        return getattr(parent, object_type)(**kwargs)
    return pydoc.locate(object_type)(**kwargs)


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
    def __init__(self, mel_df, train_ds=False):
        self.sounds_clean = mel_df['sounds_clean'].values
        self.sounds_noisy = mel_df['sounds_noisy'].values
        self.n_samples = len(self.sounds_clean)
        if train_ds:
            self.transform = object_from_dict(train_aug)
        else:
            self.transform = object_from_dict(test_aug)

    # Indexing dataset[index]
    def __getitem__(self, index):
        sample_clean = self.sounds_clean[index]
        sample_noisy = self.sounds_noisy[index]
        sample = np.append(sample_clean[None, :, :], sample_noisy[None, :, :],
                           axis=0)
        sample = torch.tensor(sample).type(torch.FloatTensor)
        sample = self.transform(image=sample)["image"]
        sample_clean = sample[0]
        sample_noisy = sample[1]

        return sample_clean[None, :, :], sample_noisy[None, :, :]

    # len(dataset)
    def __len__(self):
        return self.n_samples


def get_dataloader(train_path: Path, val_path: Path):
    mel_df = get_mel_df(train_path)
    train_set = MelDataset(mel_df, train_ds=True)

    mel_df = get_mel_df(val_path)
    val_set = MelDataset(mel_df, train_ds=False)

    train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_set, batch_size=BATCH_SIZE, shuffle=True)

    return train_loader, val_loader
