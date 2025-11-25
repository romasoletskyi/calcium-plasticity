import copy
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from args import SimulationArgs
from config import FigConfig


def run(
    args: SimulationArgs, num_runs: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert isinstance(args.calcium.D, torch.Tensor)
    assert isinstance(args.synapse.down_init_probability, torch.Tensor)
    assert isinstance(args.synapse.theta_p, torch.Tensor)
    assert isinstance(args.synapse.theta_d, torch.Tensor)
    assert isinstance(args.neuron.spike_rate, torch.Tensor)
    assert isinstance(args.neuron.pre_post_delay, torch.Tensor)

    calcium = torch.zeros((num_runs, len(args.calcium)))
    rho_init = (
        torch.arange(0, 1, 1 / num_runs)[:, None]
        >= args.synapse.down_init_probability[None, :]
    )
    rho = rho_init.float()

    num_steps = int(args.simulation_time // args.step_time)
    spike_step_period = (1 / (args.step_time * args.neuron.spike_rate)).long()
    pre_spike_shift = (args.calcium.D // args.step_time).long()
    post_spike_shift = (args.neuron.pre_post_delay // args.step_time).long()

    for step_idx in tqdm(range(num_steps)):
        is_pre_spike = (step_idx - pre_spike_shift) % spike_step_period == 0
        is_post_spike = (step_idx - post_spike_shift) % spike_step_period == 0
        dcalcium = (
            -calcium * args.step_time / args.calcium.tau_ca
            + args.calcium.c_pre * is_pre_spike
            + args.calcium.c_post * is_post_spike
        )

        drho = (
            # deterministic part
            (
                -rho * (1 - rho) * (args.synapse.rho_star - rho)
                + args.synapse.gamma_p * (1 - rho) * (calcium > args.synapse.theta_p)
                - args.synapse.gamma_d * rho * (calcium > args.synapse.theta_d)
            )
            * (args.step_time / args.synapse.tau)
        ) + (
            # noise part
            args.synapse.sigma
            * torch.normal(0, 1, size=rho.shape)
            * (calcium > torch.minimum(args.synapse.theta_p, args.synapse.theta_d))
            * (args.step_time / args.synapse.tau) ** (1 / 2)
        )

        calcium += dcalcium
        rho += drho

    rho_final = rho > args.synapse.rho_star
    return rho_init, rho, rho_final


def main(
    config_name: str,
    run_name: str,
    pre_post_delay_min: float = -100,
    pre_post_delay_max: float = 100.1,
    pre_post_delay_step: float = 5,
    num_runs: int = 100,
) -> None:
    run_path = Path(__file__).parent / "runs" / run_name
    run_path.mkdir(exist_ok=True, parents=True)

    default_args = FigConfig[config_name]
    args_list: list[SimulationArgs] = []

    pre_post_delays = np.arange(
        pre_post_delay_min, pre_post_delay_max, pre_post_delay_step
    )
    for pre_post_delay in pre_post_delays:
        args = copy.deepcopy(default_args)
        args.neuron.pre_post_delay = pre_post_delay
        args_list.append(args)

    args = SimulationArgs.batch(args_list)
    rho_init, rho, rho_final = run(args, num_runs)
    torch.save(rho_init, run_path / "rho_init.pt")
    torch.save(rho, run_path / "rho.pt")
    torch.save(rho_final, run_path / "rho_final.pt")

    init_strength = torch.mean(
        1 + rho_init * (args.synapse.up_down_strength_ratio - 1), dim=0
    )
    final_strength = torch.mean(
        1 + rho_final * (args.synapse.up_down_strength_ratio - 1), dim=0
    )
    std_strength = (args.synapse.up_down_strength_ratio - 1) * torch.sqrt(
        (
            torch.std(rho_final[rho_init].float(), dim=0) ** 2
            * rho_init.float().mean(dim=0)
            + torch.std(rho_final[~rho_init].float(), dim=0) ** 2
            * (~rho_init).float().mean(dim=0)
        )
        / num_runs
    )

    plt.plot(pre_post_delays, final_strength / init_strength, "o-", color="b")
    plt.fill_between(
        pre_post_delays,
        (final_strength - std_strength) / init_strength,
        (final_strength + std_strength) / init_strength,
        color="b",
        alpha=0.3,
    )

    plt.xlabel("post-pre delay, ms")
    plt.ylabel("synaptic strength change")

    plt.savefig(run_path / f"fig2_{config_name}.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    """
    python -m simulation --config_name DP --run_name dt0.1_runs100
    """
    fire.Fire(main)
