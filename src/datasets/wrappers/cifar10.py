import os

import numpy as np
import pickle
import matplotlib.pyplot as plt


class CIFAR10:
    """
    The CIFAR-10 dataset.
    """
    class Data:
        """
        Constructs the dataset.
        """
        def __init__(self, x, l, flip: bool, dequantize: bool, rng: np.random.RandomState):

            D = int(x.shape[1] / 3)                            # number of pixels
            if dequantize:
                x = (x + rng.rand(*x.shape)) / 256.0
            else:
                x = x.astype(np.int64)
            x = self._flip_augmentation(x) if flip else x      # flip
            self.x = x                                         # pixel values
            self.r = self.x[:, :D]                             # red component
            self.g = self.x[:, D:2*D]                          # green component
            self.b = self.x[:, 2*D:]                           # blue component
            self.y = np.hstack([l, l]) if flip else l          # numeric labels
            self.N = self.x.shape[0]                           # number of datapoints

        @staticmethod
        def _flip_augmentation(x):
            """
            Augments dataset x with horizontal flips.
            """
            D = int(x.shape[1] / 3)
            I = int(np.sqrt(D))
            r = x[:,    :D].reshape([-1, I, I])[:, :, ::-1].reshape([-1, D])
            g = x[:, D:2*D].reshape([-1, I, I])[:, :, ::-1].reshape([-1, D])
            b = x[:,  2*D:].reshape([-1, I, I])[:, :, ::-1].reshape([-1, D])
            x_flip = np.hstack([r, g, b])
            return np.vstack([x, x_flip])

    def __init__(self, path: str, flip: bool = False, dequantize: bool = True):
        rng = np.random.RandomState(42)
        path = os.path.join(path, 'cifar10')

        # load train batches
        x = []
        l = []
        for i in range(1, 6):
            with open(os.path.join(path, f'data_batch_{i}'), 'rb') as f:
                d = pickle.load(f, encoding='latin1')
                x.append(d['data'])
                l.append(d['labels'])
        x = np.concatenate(x, axis=0)
        l = np.concatenate(l, axis=0)

        # use part of the train batches for validation
        split = int(0.9 * x.shape[0])
        self.trn = self.Data(x[:split], l[:split], flip, dequantize, rng)
        self.val = self.Data(x[split:], l[split:], False, dequantize, rng)

        # load test batch
        with open(os.path.join(path, 'test_batch'), 'rb') as f:
            d = pickle.load(f, encoding='latin1')
            x = d['data']
            l = np.array(d['labels'])
            self.tst = self.Data(x, l, False, dequantize, rng)

        self.num_features = self.trn.x.shape[1]
        self.image_shape = tuple([3] + [int(np.sqrt(self.num_features / 3))] * 2)
