import argparse
from pathlib import Path
import numpy as np
from src.utils import save_images
from src.settings import *
from src.model import UMelResNet


def data_preparation(x: np.array):
    x_tensor = torch.tensor(x[None, :, :]).type(torch.FloatTensor)
    return x_tensor


def load_model(path: str):
    saved_model = UMelResNet().to(DEVICE)
    saved_model.load_state_dict(torch.load(path))
    saved_model.train(False)
    return saved_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    required = parser.add_argument_group('required arguments')
    required.add_argument("-m", "--mode", help="classification or denoising", type=str, required=True)
    required.add_argument("-w", "--weights", help="path to model's weights", type=str, required=True)
    required.add_argument("-f", "--file", help="path to .npy file", type=str, required=True)
    parser.add_argument("-o", "--out", help="path to an output folder", type=str)
    args = parser.parse_args()

    mode = args.mode
    file_path = Path(args.file)
    weights_path = args.weights

    output_path = args.out
    if output_path is not None:
        output_path = Path(output_path)

    data = np.load(file_path)
    data_tensor = data_preparation(data)
    data_tensor = data_tensor.to(DEVICE)
    model = load_model(weights_path)

    if mode == 'classification':
        is_noisy = model.is_noisy(data_tensor)
        sound = 'noisy' if is_noisy else 'clean'
        print(sound)

    elif mode == 'denoising':
        out_data = model.denoising(data_tensor).detach().cpu().numpy()
        if output_path is not None:
            save_images([data, out_data], str(output_path / 'mel.png'))
            out_data = out_data.T
            np.save(str(output_path / 'out.npy'), out_data)
        else:
            print('You did not pass the output directory')
    else:
        print('You chose incorrect mode')



