from pathlib import Path
from struct import unpack

import numpy as np


def read_mnist(run_path: Path, training: bool) -> tuple[np.ndarray, np.ndarray]:
    tag = "train" if training else "t10k"
    mnist_path = run_path / "MNIST" / "raw"

    images = open(mnist_path / ("%s-images-idx3-ubyte" % tag), "rb")
    images.read(4)
    n_images = unpack(">I", images.read(4))[0]
    n_rows = unpack(">I", images.read(4))[0]
    n_cols = unpack(">I", images.read(4))[0]
    assert n_rows == n_cols and n_rows * n_cols == 784, (n_rows, n_cols)

    labels = open(mnist_path / ("%s-labels-idx1-ubyte" % tag), "rb")
    labels.read(4)
    n_labels = unpack(">I", labels.read(4))[0]
    assert n_images == n_labels, (n_images, n_labels)

    x = np.frombuffer(images.read(), dtype=np.uint8).reshape(n_images, -1) / 8.0
    y = np.frombuffer(labels.read(), dtype=np.uint8)
    return x, y
