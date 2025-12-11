import logging

import fire
import numpy as np
import torch
from tqdm import tqdm

from network.args import EvalArgs, SampleArgs
from network.config import NetworkConfig, SampleConfig
from network.model import Network, NeuronGroup, PoissonGroup, build_network
from network.utils import read_mnist
from utils import get_run_path

logger = logging.getLogger(__file__)


def show_sample(
    args: SampleArgs,
    network: Network,
    sample: torch.Tensor,
    step_time: float,
    training: bool,
) -> tuple[torch.Tensor, list]:
    """
    Args:
        args: how sample is shown
        network: pg_inp -> ng_exc <-> ng_inh
        sample: flattened image
        step_time: simulation step time
        training: if network weights are training

    Returns:
        spike_count (torch.Tensor) : number of spikes during stimulation per each neuron in ng_exc
    """

    # TODO: @roman how do discern neuron groups? names?
    assert isinstance(network.neurons[0], PoissonGroup)
    assert isinstance(network.neurons[1], NeuronGroup)
    assert isinstance(network.neurons[2], NeuronGroup)
    intensity = args.starting_intensity

    voltage: list[tuple[torch.Tensor, torch.Tensor]] = []

    while True:
        # time is computed in ms but sample intensity is in Hz
        network.neurons[0].rate = sample * intensity / 1000
        spike_count = torch.zeros(network.neurons[1].size)

        for _ in range(int(args.stimulation_time // step_time)):
            network.step(step_time, training)
            voltage.append((network.neurons[1].voltage, network.neurons[2].voltage))
            spike_count += network.neurons[1].spike_mask

        for _ in range(int(args.rest_time // step_time)):
            network.step(step_time, training)

        logger.info(f"Fired neurons {torch.arange(len(spike_count))[spike_count > 0]}")
        if spike_count.sum() >= args.spike_threshold:
            logger.info(
                f"spike count {spike_count.sum()} >= {args.spike_threshold=} - stop"
            )
            return spike_count, voltage
        else:
            intensity += args.intensity_increase
            logger.info(
                f"spike count {spike_count.sum()} < {args.spike_threshold=} - retry with {intensity=}"
            )


def eval(run_name: str) -> None:
    run_path = get_run_path(run_name)
    data_path = run_path / "data"

    args = EvalArgs(
        num_classes=10,
        num_samples=1000,
        sample=SampleConfig["base"],
    )
    step_time = 0.1

    network = build_network(
        args=NetworkConfig["base"],
        ckpt_path=run_path / "data",
    )

    confusion = np.zeros((args.num_classes, args.num_classes))
    assign = np.load(data_path / "assign.npy")
    groups = [np.where(assign == i)[0] for i in range(args.num_classes)]

    images, labels = read_mnist(run_path, False)
    indices = np.arange(args.num_samples)
    np.random.shuffle(indices)

    for idx in tqdm(indices):
        spike_count = show_sample(
            args=args.sample,
            network=network,
            sample=torch.tensor(images[idx]),
            step_time=step_time,
            training=False,
        )
        guess = np.argmax([spike_count[group].mean().item() for group in groups])
        confusion[labels[idx], guess] += 1

    confusion = confusion / confusion.sum(axis=1)[:, None]
    print(f"Accuracy: {(np.trace(confusion) / np.sum(confusion)):.3f}")
    print(np.around(confusion, 2))
    np.save(data_path / "confussion_eval.npy", confusion)


if __name__ == "__main__":
    """
    python -m network.eval --run_name brian_repro
    """
    fire.Fire(eval)
