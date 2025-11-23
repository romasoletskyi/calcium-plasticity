import copy
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from args import SimulationArgs
from config import FigConfig
from utils import extract_unique


def run(
    args_list: list[SimulationArgs], num_runs: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # scalar parameters
    simulation_time = extract_unique([args.simulation_time for args in args_list])
    step_time = extract_unique([args.step_time for args in args_list])
    spike_rate = extract_unique([args.spike_rate for args in args_list])

    # vector parameters
    tau_ca = np.array([args.calcium.tau_ca for args in args_list])
    c_pre = np.array([args.calcium.c_pre for args in args_list])
    c_post = np.array([args.calcium.c_post for args in args_list])
    D = np.array([args.calcium.D for args in args_list])

    tau = np.array([args.synapse.tau for args in args_list])
    rho_star = np.array([args.synapse.rho_star for args in args_list])
    gamma_p = np.array([args.synapse.gamma_p for args in args_list])
    gamma_d = np.array([args.synapse.gamma_d for args in args_list])
    theta_p = np.array([args.synapse.theta_p for args in args_list])
    theta_d = np.array([args.synapse.theta_d for args in args_list])
    sigma = np.array([args.synapse.sigma for args in args_list])

    pre_post_delay = np.array([args.pre_post_delay for args in args_list])
    down_init_probability = np.array([args.down_init_probability for args in args_list])

    # starting values and logging
    calcium = np.zeros((num_runs, len(args_list)))
    rho_init = np.arange(0, 1, 1 / num_runs)[:, None] >= down_init_probability[None, :]
    rho = np.asarray(rho_init, dtype=float)

    rng = np.random.default_rng()
    time = np.arange(0, simulation_time, step_time)

    for t in tqdm(time):
        pre_spike_phase = np.modf((t - D) * spike_rate)[0]
        is_pre_spike = (0 <= pre_spike_phase) & (
            pre_spike_phase < step_time * spike_rate
        )

        post_spike_phase = np.modf((t - pre_post_delay) * spike_rate)[0]
        is_post_spike = (0 <= post_spike_phase) & (
            post_spike_phase < step_time * spike_rate
        )

        dcalcium = (
            -calcium * step_time / tau_ca
            + c_pre * is_pre_spike
            + c_post * is_post_spike
        )

        drho = (
            # deterministic part
            (
                -rho * (1 - rho) * (rho_star - rho)
                + gamma_p * (1 - rho) * (calcium > theta_p)
                - gamma_d * rho * (calcium > theta_d)
            )
            * (step_time / tau)
        ) + (
            # noise part
            sigma
            * rng.normal(size=rho.shape)
            * (calcium > np.minimum(theta_d, theta_d))
            * (step_time / tau) ** (1 / 2)
        )

        calcium += dcalcium
        rho += drho

    rho_final = rho > rho_star
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
        args.pre_post_delay = pre_post_delay
        args_list.append(args)

    rho_init, rho, rho_final = run(args_list, num_runs)
    np.save(run_path / "rho_init.npy", rho_init)
    np.save(run_path / "rho.npy", rho)
    np.save(run_path / "rho_final.npy", rho_final)

    up_down_strength_ratio = np.array(
        [args.synapse.up_down_strength_ratio for args in args_list]
    )

    init_strength = np.mean(1 + rho_init * (up_down_strength_ratio - 1), axis=0)
    final_strength = np.mean(1 + rho_final * (up_down_strength_ratio - 1), axis=0)
    std_strength = (up_down_strength_ratio - 1) * np.sqrt(
        (
            np.std(rho_final[rho_init]) ** 2 * np.mean(rho_init)
            + np.std(rho_final[~rho_init]) ** 2 * np.mean(~rho_init)
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
