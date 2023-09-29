import os
import numpy as np
import gzip
import pickle


class MNIST:
    """
    The MNIST dataset of handwritten digits.
    """
    class Data:
        """
        Constructs the dataset.
        """

        def __init__(self, data, dequantize: bool, rng: np.random.RandomState):
            self.x = data[0]
            if dequantize:
                self.x = self.x + rng.rand(*self.x.shape) / 256.0
            else:
                self.x = (self.x * 256.0).astype(np.int64)
            self.y = data[1]                                               # numeric labels
            self.N = self.x.shape[0]

    def __init__(self, path: str, dequantize: bool = True):
        # load dataset
        with gzip.open(os.path.join(path, 'mnist', 'mnist.pkl.gz'), 'rb') as f:
            trn, val, tst = pickle.load(f, encoding='latin1')

        rng = np.random.RandomState(42)
        self.trn = self.Data(trn, dequantize, rng)
        self.val = self.Data(val, dequantize, rng)
        self.tst = self.Data(tst, dequantize, rng)

        self.num_features = self.trn.x.shape[1]
        self.image_shape = tuple([1] + [int(np.sqrt(self.num_features))] * 2)
