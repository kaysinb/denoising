import matplotlib.pyplot as plt
import numpy as np
import torch


class TrainingLogging:
    def __init__(self):
        self.logs = {}

    def add(self, key, value, step):
        if key not in self.logs:
            self.logs[key] = {'values': [], 'steps': []}
        self.logs[key]['values'].append(value)
        self.logs[key]['steps'].append(step)

    def __getitem__(self, key):
        return self.logs[key]['steps'], self.logs[key]['values']

    def get_keys(self):
        return list(self.logs.keys())

    def plot(self, *keys, path=None):
        fig, ax = plt.subplots(figsize=(25, 8))
        existed_keys = self.get_keys()
        for key in keys:
            if key in existed_keys:
                ax.plot(*self[key], label=key)
            else:
                print(f"key - '{key}' does not exist")
        plt.xlabel("step")
        plt.ylabel("value")
        plt.legend(loc='best')

        if path is not None:
            plt.savefig(path)


def save_images(images, path):
    if len(images) == 2:
        descriptions = ['before', 'after']
        left_shift = [13, 10]
    else:
        descriptions = ['clean before', 'clean after', 'noisy before', 'noisy after']
        left_shift = [25, 22, 25, 22]

    np_images = None
    for img in images:
        if type(img) is torch.Tensor:
            img = img.detach().cpu().numpy()[0]

        if np_images is not None:
            np_images = np.vstack((np_images, img))
        else:
            np_images = img

    fig, ax = plt.subplots(figsize=(16, 16))
    plt.imshow(np_images)
    for i, description in enumerate(descriptions):
        plt.text(left_shift[i], 73 + 80 * i, description, size=20,
                 ha="center", va="center",
                 bbox=dict(boxstyle="round",
                           ec=(1., 0.5, 0.5),
                           fc=(1., 0.8, 0.8),
                           )
                 )
    plt.savefig(path)
