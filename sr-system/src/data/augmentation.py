import numpy as np


class Augmentor:
    def __init__(
        self,
        hflip=True,
        vflip=True,
        rotation=True,
        color_jitter=False
    ):
        self.hflip = hflip
        self.vflip = vflip
        self.rotation = rotation
        self.color_jitter = color_jitter

    def augment(self, hr, lr):
        # ---- Single random state
        if self.hflip and np.random.rand() > 0.5:
            hr = hr[:, ::-1, :]
            lr = lr[:, ::-1, :]

        if self.vflip and np.random.rand() > 0.5:
            hr = hr[::-1, :, :]
            lr = lr[::-1, :, :]

        if self.rotation:
            k = np.random.randint(0, 4)
            hr = np.rot90(hr, k, axes=(0, 1))
            lr = np.rot90(lr, k, axes=(0, 1))

        # ---- Color jitter (optional, HR only)
        if self.color_jitter:
            brightness = np.random.uniform(-20, 20)
            contrast = np.random.uniform(0.8, 1.2)

            hr = np.clip(hr * contrast + brightness, 0, 255)

        return hr, lr