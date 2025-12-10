import fire
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def main() -> None:
    simulation_time = 600_000
    step_time = 1
    spike_rate = 0.005
    num_runs = 1000

    pre_post_delay_min = -100
    pre_post_delay_max = 100.1
    pre_post_delay_step = 5

    tau_pre_trace = 20
    tau_post_trace_update_at_pre = 20
    tau_post_trace_update_at_post = 40

    pre_post_delay = np.arange(
        pre_post_delay_min, pre_post_delay_max, pre_post_delay_step
    )

    num_steps = int(simulation_time // step_time)
    spike_step_period = int(1 / (step_time * spike_rate))
    pre_spike_shift = np.zeros_like(pre_post_delay, dtype=int)
    post_spike_shift = np.asarray(pre_post_delay // step_time, dtype=int)

    weights = 0.3 * np.random.random((len(pre_post_delay), num_runs))
    pre_trace = np.zeros_like(weights)
    post_trace_update_at_pre = np.zeros_like(weights)  # post1
    post_trace_update_at_post = np.zeros_like(weights)  # post2

    for step_idx in tqdm(range(num_steps)):
        is_pre_spike = (step_idx - pre_spike_shift) % spike_step_period == 0
        is_post_spike = (step_idx - post_spike_shift) % spike_step_period == 0

        # on pre
        pre_trace[is_pre_spike] = 1.0
        weights[is_pre_spike] = np.clip(
            weights[is_pre_spike] - 0.0001 * post_trace_update_at_pre[is_pre_spike],
            0,
            1,
        )

        # on post
        weights[is_post_spike] = np.clip(
            weights[is_post_spike]
            + 0.01
            * pre_trace[is_post_spike]
            * post_trace_update_at_post[is_post_spike],
            0,
            1,
        )
        post_trace_update_at_post[is_post_spike] = 1
        post_trace_update_at_pre[is_post_spike] = 1

        # regular
        pre_trace -= pre_trace * (step_time / tau_pre_trace)
        post_trace_update_at_pre -= post_trace_update_at_pre * (
            step_time / tau_post_trace_update_at_pre
        )
        post_trace_update_at_post -= post_trace_update_at_post * (
            step_time / tau_post_trace_update_at_post
        )

    plt.plot(pre_post_delay, np.mean(weights, axis=1))
    plt.show()


if __name__ == "__main__":
    fire.Fire(main)
