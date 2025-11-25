from dataclasses import dataclass, fields
from typing import Type, TypeVar

import torch

from utils import extract_unique

T = TypeVar("T", bound="TensorArgs")
ArgType = float | torch.Tensor


@dataclass
class TensorArgs:
    """
    Dataclass mixin for simulation parameters.
    Fields may be float or Tensor. `batchify` creates a new instance
    where all fields are batched tensors of shape [N].
    """

    @classmethod
    def batch(cls: Type[T], args_list: list[T]) -> T:
        assert len(args_list) > 0
        batched = {}
        for f in fields(cls):
            values = [getattr(p, f.name) for p in args_list]
            tensor = torch.tensor(values, dtype=torch.float32)
            batched[f.name] = tensor
        return cls(**batched)

    def __len__(self) -> int:
        lengths: list[int] = []
        for f in fields(self):
            value = getattr(self, f.name)
            assert isinstance(value, ArgType)
            if isinstance(value, float):
                lengths.append(1)
            else:
                lengths.append(len(value))

        lengths = list(set(lengths))
        assert len(lengths) == 1
        return lengths[0]


@dataclass
class CalciumArgs(TensorArgs):
    r"""
    Parameters of 3.1.1 Simplified calcium model (1).
    \frac{dc}{dt} =
        - \frac{c}{\tau_{Ca}}
        + C_{pre} \sum_i \delta(t - t_i - D)
        + C_{post} \sum_j \delta(t - t_j)
    """

    # calcium relaxation time (ms)
    tau_ca: ArgType

    # pre/post calcium aplitudes
    c_pre: ArgType
    c_post: ArgType

    # time delay etween presynaptic spike and calcium transient (ms)
    D: ArgType


@dataclass
class SynapticArgs(TensorArgs):
    r"""Parameters of main equation (1).
    \tau \frac{d\rho}{dt} =
        - \rho (1 - \rho) (\rho_\star - \rho)
        + \gamma_p (1 - \rho) \Theta[c - \theta_p]
        - \gamma_d \rho \Theta[c - \theta_d]
        + \sigma \sqrt{\tau} \Theta[c - \min(\theta_d, \theta_p)] \eta
    """

    # synaptic efficacy relaxation time (ms)
    tau: ArgType

    # unstable synaptic weight (0 and 1 are stable weights)
    rho_star: ArgType

    # potentiation/depression synaptic change rate
    gamma_p: ArgType
    gamma_d: ArgType

    # potentiation/depression threshold
    theta_p: ArgType
    theta_d: ArgType

    # noise strength
    sigma: ArgType

    # synaptic weight strength and init
    up_down_strength_ratio: ArgType
    down_init_probability: ArgType


@dataclass
class NeuronArgs(TensorArgs):
    spike_rate: ArgType
    pre_post_delay: ArgType


@dataclass
class SimulationArgs:
    calcium: CalciumArgs
    synapse: SynapticArgs
    neuron: NeuronArgs

    simulation_time: float
    step_time: float

    @classmethod
    def batch(
        cls: Type["SimulationArgs"], args_list: list["SimulationArgs"]
    ) -> "SimulationArgs":
        return SimulationArgs(
            calcium=CalciumArgs.batch([args.calcium for args in args_list]),
            synapse=SynapticArgs.batch([args.synapse for args in args_list]),
            neuron=NeuronArgs.batch([args.neuron for args in args_list]),
            simulation_time=extract_unique(
                [args.simulation_time for args in args_list]
            ),
            step_time=extract_unique([args.step_time for args in args_list]),
        )
