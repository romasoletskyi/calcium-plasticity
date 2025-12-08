from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch

from network.args import NetworkArgs, NeuronArgs
from network.config import NeuronConfig


class SpikingGroup(ABC):
    def __init__(self, size: int, step_time: float) -> None:
        self.size = size
        self.step_time = step_time
        self.spike_mask = torch.zeros(size, dtype=torch.bool)

    @abstractmethod
    def step(self) -> None:
        pass


class PoissonGroup(SpikingGroup):
    def __init__(self, size: int, rate: torch.Tensor, step_time: float) -> None:
        super().__init__(size, step_time)
        self.rate = rate

    def step(self) -> None:
        self.spike_mask = torch.rand(self.size) < self.rate * self.step_time


class NeuronGroup(SpikingGroup):
    def __init__(
        self, args: NeuronArgs, size: int, step_time: float, theta: torch.Tensor
    ) -> None:
        super().__init__(size, step_time)
        self.args = args

        self.voltage = args.v_rest * torch.ones(size)
        self.g_exc = torch.zeros(size)
        self.g_inh = torch.zeros(size)
        self.theta = theta
        self.timer = torch.zeros(size)

    def step(self) -> None:
        # TODO: renormalise g_exc, g_inh to be outside of tau
        dvoltage = (
            self.args.v_rest
            - self.voltage
            + self.g_exc * (self.args.v_exc_base - self.voltage)
            + self.g_inh * (self.args.v_inh_base - self.voltage)
        ) * (self.step_time / self.args.tau_membrane)
        dg_exc = -self.g_exc * (self.step_time / self.args.tau_g_exc)
        dg_inh = -self.g_inh * (self.step_time / self.args.tau_g_inh)
        dtheta = -self.theta * (self.step_time / self.args.tau_theta)

        # Don't update voltage when neuron just spiked and in refractory stage
        self.voltage += dvoltage * (self.timer > self.args.v_refractory)
        self.g_exc += dg_exc
        self.g_inh += dg_inh
        self.theta += dtheta
        self.timer += self.step_time

        # Reset voltage and increase firing threshold after spike
        self.spike_mask = (self.voltage > self.theta + self.args.theta_shift) & (
            self.timer > max(self.args.v_refractory, self.args.timer_refractory or 0.0)
        )
        assert not torch.isnan(self.voltage).any(), (self.g_exc, self.g_inh)
        self.voltage = (
            self.args.v_reset * self.spike_mask + self.voltage * ~self.spike_mask
        )
        self.theta += self.args.theta_spike * self.spike_mask


class SynapseGroup:
    def __init__(
        self,
        pre: SpikingGroup,
        post: NeuronGroup,
        exc_weight: torch.Tensor,
        inh_weight: torch.Tensor,
    ) -> None:
        self.pre = pre
        self.post = post
        self.exc_weight = exc_weight
        self.inh_weight = inh_weight

    def step(self) -> None:
        self.post.g_exc += self.exc_weight @ self.pre.spike_mask.float()
        self.post.g_inh += self.inh_weight @ self.pre.spike_mask.float()


class Network:
    def __init__(
        self,
        args: NetworkArgs,
        neurons: list[SpikingGroup],
        synapses: list[SynapseGroup],
    ) -> None:
        self.args = args
        self.neurons = neurons
        self.synapses = synapses

    def step(self) -> None:
        for neurons in self.neurons:
            neurons.step()

        for synapses in self.synapses:
            synapses.step()


def build_network(args: NetworkArgs, run_path: Path) -> Network:
    data_path = run_path / "data"
    size = args.hidden_size

    pg_inp = PoissonGroup(
        size=args.input_size,
        rate=torch.zeros(args.input_size),
        step_time=args.step_time,
    )

    theta = torch.tensor(np.load(data_path / "theta.npy"), dtype=torch.float)
    ng_exc = NeuronGroup(
        NeuronConfig["EXC"],
        size=size,
        step_time=args.step_time,
        theta=theta,
    )
    # TODO: @roman remove when train/eval is done
    ng_exc.args.theta_spike = 0.0
    ng_exc.args.tau_theta = float("inf")

    ng_inh = NeuronGroup(
        NeuronConfig["INH"],
        size=size,
        step_time=args.step_time,
        theta=torch.zeros(size),
    )

    weight = (
        torch.tensor(np.load(data_path / "weights.npy"), dtype=torch.float)
        .reshape(args.input_size, size)
        .T
    )
    syns_inp_exc = SynapseGroup(
        pre=pg_inp,
        post=ng_exc,
        exc_weight=weight,
        inh_weight=torch.zeros(size, args.input_size),
    )
    syns_exc_inh = SynapseGroup(
        pre=ng_exc,
        post=ng_inh,
        exc_weight=args.weight_exc_inh * torch.eye(size),
        inh_weight=torch.zeros(size, size),
    )
    syns_inh_exc = SynapseGroup(
        pre=ng_inh,
        post=ng_exc,
        exc_weight=torch.zeros(size, size),
        inh_weight=args.weight_inh_exc * (torch.ones(size, size) - torch.eye(size)),
    )

    return Network(
        args=args,
        neurons=[pg_inp, ng_exc, ng_inh],
        synapses=[syns_inp_exc, syns_exc_inh, syns_inh_exc],
    )
