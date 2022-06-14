import argparse
from predict import load_model, data_separation, data_aggregation
from pathlib import Path
from src.mel_dataset import get_mel_df
import numpy as np
from tqdm import tqdm
from src.settings import *


def mse(ar1, ar2):
    return np.sum((ar1-ar2)**2) / (ar1.shape[0] * ar1.shape[1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    required = parser.add_argument_group('required arguments')
    required.add_argument("-w", "--weights", help="path to model's weights", type=str, required=True)
    required.add_argument("-t", "--test_folder", help="test folder where located clean and noisy directories",
                          type=str, required=True)
    args = parser.parse_args()

    test_path = Path(args.test_folder)
    mel_df = get_mel_df(test_path)
    weights_path = args.weights
    model = load_model(weights_path)

    accuracy = 0
    mse_avg = 0

    print(f'Your device is {DEVICE}')
    print('Test started ...')
    for clean, noisy in tqdm(zip(mel_df['sounds_clean'].values, mel_df['sounds_noisy'].values), total=len(mel_df)):
        clean = clean.T
        noisy = noisy.T
        clean_tensor = data_separation(clean).to(DEVICE)
        noisy_tensor = data_separation(noisy).to(DEVICE)

        res_clean = 0 if model.is_noisy(clean_tensor) else 1
        res_noisy = 1 if model.is_noisy(noisy_tensor) else 0
        accuracy += res_clean + res_noisy

        out_tensor_clean = model.denoising(clean_tensor)
        out_tensor_noisy = model.denoising(noisy_tensor)

        out_clean = data_aggregation(out_tensor_clean, clean.shape[1])
        out_noisy = data_aggregation(out_tensor_noisy, noisy.shape[1])
        mse_avg += mse(clean, out_clean) + mse(clean, out_noisy)

    accuracy /= (len(mel_df) * 2)
    mse_avg /= (len(mel_df) * 2)
    print('Test finished!')

    print(f'test accuracy = {accuracy}')
    print(f'test mse = {mse_avg}')

