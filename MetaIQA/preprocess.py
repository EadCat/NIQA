import numpy as np
import cv2

import torch


class Preprocessor:
    def __init__(self, resize):
        self.resizer = Resize(resize)
        self.normalizer = Normalize()
        self.totensor = ToTensor()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, image: np.ndarray):
        image = self.resizer(image)
        image = self.normalizer(image)
        image = self.totensor(image)
        return image


class Normalize(object):
    def __init__(self):
        self.means = np.array([0.485, 0.456, 0.406])
        self.stds = np.array([0.229, 0.224, 0.225])

    def __call__(self, image:np.ndarray):
        image = image / 255.  # [0 ~ 255] -> [0 ~ 1]
        image[:, :, 0] = (image[:, :, 0] - self.means[0]) / self.stds[0]
        image[:, :, 1] = (image[:, :, 1] - self.means[1]) / self.stds[1]
        image[:, :, 2] = (image[:, :, 2] - self.means[2]) / self.stds[2]
        return image


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image: np.ndarray):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image).float()
        one_batch = image.unsqueeze(0)
        return one_batch


class Resize(object):
    def __init__(self, resize):
        self.h, self.w = resize

    def __call__(self, image:np.ndarray):
        image = cv2.resize(image, (self.w, self.h))
        return image

