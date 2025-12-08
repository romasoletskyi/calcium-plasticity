from dataclasses import dataclass

import fire
import torchvision
from torch.utils.data import DataLoader

from network.args import NetworkArgs
from utils import get_run_path


@dataclass
class TrainArgs:
    ckpt_freq: int

    network: NetworkArgs


def train(run_name: str) -> None:
    run_path = get_run_path(run_name)

    dataset = torchvision.datasets.MNIST(
        root=str(run_path),
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )

    # we don't shuffle for consistency with brian2.py
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)


if __name__ == "__main__":
    """
    python -m network.stdp train --run_name brian_repro
    """
    fire.Fire(train)
