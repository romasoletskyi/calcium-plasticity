import fire
import numpy as np
import torch
from tqdm import tqdm

from network.args import EvalArgs, NetworkArgs
from network.brian2 import read_mnist
from network.model import Network, PoissonGroup, build_network
from utils import get_run_path


def show_sample(args: EvalArgs, network: Network, sample: torch.Tensor) -> torch.Tensor:
    """
    Args:
        network (Network) : pg_inp -> ng_exc <-> ng_inh
        sample (torch.Tensor) : flattened image
        intensity (float) : pixel intensity multiplier

    Returns:
        spike_count (torch.Tensor) : number of spikes during stimulation per each neuron in ng_exc
    """

    # TODO: @roman how do discern neuron groups? names?
    assert isinstance(network.neurons[0], PoissonGroup)
    intensity = args.starting_intensity

    while True:
        # time is computed in ms but sample intensity is in Hz
        network.neurons[0].rate = sample * intensity / 1000
        spike_count = torch.zeros(network.neurons[1].size)

        for _ in range(int(args.stimulation_time // network.args.step_time)):
            network.step()
            spike_count += network.neurons[1].spike_mask

        for _ in range(int(args.rest_time // network.args.step_time)):
            network.step()

        if spike_count.sum() >= args.spike_threshold:
            return spike_count
        else:
            intensity += args.intensity_increase


def eval(run_name: str) -> None:
    run_path = get_run_path(run_name)
    data_path = run_path / "data"

    args = EvalArgs(
        num_classes=10,
        num_samples=1,
        stimulation_time=350,
        rest_time=150,
        spike_threshold=0,
        starting_intensity=2,
        intensity_increase=1,
    )

    network = build_network(
        args=NetworkArgs(
            input_size=784,
            hidden_size=400,
            step_time=0.1,
            weight_exc_inh=10.4,
            weight_inh_exc=17.0,
        ),
        run_path=run_path,
    )

    confusion = np.zeros((args.num_classes, args.num_classes))
    assign = np.load(data_path / "assign.npy")
    groups = [np.where(assign == i)[0] for i in range(args.num_classes)]

    images, labels = read_mnist(run_path, False)
    indices = np.arange(args.num_samples)
    np.random.shuffle(indices)

    for idx in tqdm(indices):
        spike_count = show_sample(args, network, torch.tensor(images[idx]))
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
