from dataclasses import dataclass

import fire
import torch
import torchvision
from tqdm import tqdm

from network.args import NetworkArgs, SampleArgs
from network.config import NetworkConfig, SampleConfig
from network.eval import show_sample
from network.model import build_network
from network.utils import read_mnist
from utils import get_run_path


@dataclass
class TrainArgs:
    max_steps: int
    ckpt_freq: int

    network: NetworkArgs
    sample: SampleArgs
    step_time: float


def train(run_name: str) -> None:
    run_path = get_run_path(run_name)

    args = TrainArgs(
        max_steps=10,
        ckpt_freq=10,
        network=NetworkConfig["base"],
        sample=SampleConfig["base"],
        step_time=0.1,
    )

    # download MNIST
    torchvision.datasets.MNIST(
        root=str(run_path),
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )

    images, labels = read_mnist(run_path, True)
    n_samples = images.shape[0]

    network = build_network(args.network, run_path)

    for idx in tqdm(range(args.max_steps)):
        idx = idx % n_samples
        show_sample(
            args=args.sample,
            network=network,
            sample=torch.tensor(images[idx]),
            step_time=args.step_time,
            training=True,
        )


if __name__ == "__main__":
    """
    python -m network.stdp train --run_name network_calcium
    """
    fire.Fire(train)
