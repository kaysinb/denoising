import argparse
from pathlib import Path
import numpy as np
from src.utils import save_images
from src.settings import *
from src.mel_dataset import extend_sample
from src.model import UMelResNet


def data_separation(x: np.array):
    x_tensor = []
    slice_num = x.shape[1]//T_SIZE
    for i in range(slice_num):
        tensor = torch.tensor(x[:, i*T_SIZE:(i+1)*T_SIZE][None, :, :]).type(torch.FloatTensor)
        x_tensor.append(tensor)

    tail_start = slice_num * T_SIZE
    if tail_start < x.shape[1]:
        tail = x[:, tail_start:]
        tail = torch.tensor(extend_sample(tail)[None, :, :]).type(torch.FloatTensor)
        x_tensor.append(tail)
    x_tensor = torch.stack(x_tensor)
    return x_tensor


def data_aggregation(x_tensor: torch.tensor, orig_data_len: int):
    output = None
    x_tensor = x_tensor.detach().cpu().numpy()
    for i in range(x_tensor.shape[0]):
        if output is None:
            output = x_tensor[i][0]
        else:
            output = np.hstack((output, x_tensor[i][0]))
    output = output[:, :orig_data_len]
    return output


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

    data = np.load(file_path).T
    data_tensor = data_separation(data)
    data_tensor = data_tensor.to(DEVICE)
    model = load_model(weights_path)

    if mode == 'classification':
        is_noisy = model.is_noisy(data_tensor)
        sound = 'noisy' if is_noisy else 'clean'
        print(sound)

    elif mode == 'denoising':
        out_tensor = model.denoising(data_tensor)
        out_data = data_aggregation(out_tensor, data.shape[1])
        if output_path is not None:
            save_images([data, out_data], str(output_path / 'mel.png'))
            out_data = out_data.T
            np.save(str(output_path / 'out.npy'), out_data)
        else:
            print('You did not pass the output directory')
    else:
        print('You chose incorrect mode')



