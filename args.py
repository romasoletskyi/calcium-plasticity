import dataclasses


@dataclasses.dataclass
class CalciumArgs:
    r"""
    Parameters of 3.1.1 Simplified calcium model (1).
    \frac{dc}{dt} =
        - \frac{c}{\tau_{Ca}}
        + C_{pre} \sum_i \delta(t - t_i - D)
        + C_{post} \sum_j \delta(t - t_j)
    """

    # calcium relaxation time (ms)
    tau_ca: float

    # pre/post calcium aplitudes
    c_pre: float
    c_post: float

    # time delay etween presynaptic spike and calcium transient (ms)
    D: float


@dataclasses.dataclass
class SynapticArgs:
    r"""Parameters of main equation (1).
    \tau \frac{d\rho}{dt} =
        - \rho (1 - \rho) (\rho_\star - \rho)
        + \gamma_p (1 - \rho) \Theta[c - \theta_p]
        - \gamma_d \rho \Theta[c - \theta_d]
        + \sigma \sqrt{\tau} \Theta[c - \min(\theta_d, \theta_p)] \eta
    """

    # synaptic efficacy relaxation time (ms)
    tau: float

    # unstable synaptic weight (0 and 1 are stable weights)
    rho_star: float

    # potentiation/depression synaptic change rate
    gamma_p: float
    gamma_d: float

    # potentiation/depression threshold
    theta_p: float
    theta_d: float

    # noise strength
    sigma: float

    # synaptic weight strength
    up_down_strength_ratio: float


@dataclasses.dataclass
class SimulationArgs:
    calcium: CalciumArgs
    synapse: SynapticArgs

    simulation_time: float
    step_time: float

    spike_rate: float
    pre_post_delay: float

    down_init_probability: float
