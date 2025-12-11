import logging
import sys
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

logger = logging.getLogger(__file__)


@dataclass
class TrainArgs:
    max_steps: int
    ckpt_freq: int

    network: NetworkArgs
    sample: SampleArgs
    step_time: float


def train(run_name: str) -> None:
    logging.basicConfig(
        stream=sys.stdout,
        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    run_path = get_run_path(run_name)
    log_path = run_path / "logs"
    log_path.mkdir(exist_ok=True, parents=True)

    args = TrainArgs(
        max_steps=40,
        ckpt_freq=1,
        network=NetworkConfig["calcium"],
        sample=SampleConfig["base"],
        step_time=0.25,
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

    network = build_network(args.network, None)
    for idx in tqdm(range(args.max_steps)):
        idx = idx % n_samples
        _, voltage = show_sample(
            args=args.sample,
            network=network,
            sample=torch.tensor(images[idx]),
            step_time=args.step_time,
            training=True,
        )

        torch.save(voltage, log_path / f"voltage_{idx:08d}.pt")
        if (idx + 1) % args.ckpt_freq == 0:
            ckpt_path = run_path / "checkpoints" / f"checkpoint_{idx:08d}"
            ckpt_path.mkdir(exist_ok=True, parents=True)
            weights_path = ckpt_path / "weights.pt"

            torch.save(network.synapses[0].exc_weight, weights_path)
            logger.info(f"Saved weights to {weights_path}")


if __name__ == "__main__":
    """
    python -m network.train --run_name network_calcium
    """
    fire.Fire(train)
