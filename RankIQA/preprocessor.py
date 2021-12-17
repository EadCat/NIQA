import numpy as np
import cv2

import torch


class Preprocessor:
    def __init__(self, patch_num):
        self.totensor = ToTensor()
        self.patch_num = patch_num

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, image: np.ndarray):
        patch_list = []
        for _ in range(self.patch_num):
            patch = Cropper(image)
            patch = self.totensor(patch)
            patch_list.append(patch)
        return torch.stack(patch_list)  # [Batch(Patch) x Channel x Height x Width]


def Cropper(image: np.ndarray):
    h, w, _ = image.shape
    w_p = np.random.randint(w - 224)
    h_p = np.random.randint(h - 224)
    patch = image[h_p: (h_p + 224), w_p: (w_p + 224), :]
    return patch


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image: np.ndarray):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        image = image.transpose((2, 0, 1)) / 255.
        image = torch.from_numpy(image).float()
        return image

