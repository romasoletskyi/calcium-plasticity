from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch

from network.args import NetworkArgs, NeuronArgs
from network.config import NeuronConfig


class SpikingGroup(ABC):
    def __init__(self, size: int) -> None:
        self.size = size
        self.spike_mask = torch.zeros(size, dtype=torch.bool)

    @abstractmethod
    def step(self, step_time: float, training: bool) -> None:
        pass


class PoissonGroup(SpikingGroup):
    def __init__(self, size: int, rate: torch.Tensor) -> None:
        super().__init__(size)
        self.rate = rate

    def step(self, step_time: float, training: bool) -> None:
        self.spike_mask = torch.rand(self.size) < self.rate * step_time


class NeuronGroup(SpikingGroup):
    def __init__(self, args: NeuronArgs, size: int, theta: torch.Tensor) -> None:
        super().__init__(size)
        self.args = args

        self.voltage = args.v_rest * torch.ones(size)
        self.g_exc = torch.zeros(size)
        self.g_inh = torch.zeros(size)
        self.theta = theta
        self.timer = self.args.spike_refractory_threshold * torch.ones(size)

    def step(self, step_time: float, training: bool) -> None:
        dvoltage = (
            self.args.v_rest
            - self.voltage
            + self.g_exc * (self.args.v_exc_base - self.voltage)
            + self.g_inh * (self.args.v_inh_base - self.voltage)
        ) * (step_time / self.args.tau_membrane)
        dg_exc = -self.g_exc * (step_time / self.args.tau_g_exc)
        dg_inh = -self.g_inh * (step_time / self.args.tau_g_inh)

        # Don't update voltage when neuron just spiked and in refractory stage
        self.voltage += dvoltage * (self.timer > self.args.v_refractory)
        self.g_exc += dg_exc
        self.g_inh += dg_inh
        self.timer += step_time

        if self.args.threshold_adaptation is not None and training:
            self.theta -= self.theta * (
                step_time / self.args.threshold_adaptation.tau_theta
            )

        # Reset voltage and increase firing threshold after spike
        self.spike_mask = (self.voltage > self.args.v_threshold + self.theta) & (
            self.timer > self.args.spike_refractory_threshold
        )
        self.voltage = (
            self.args.v_reset * self.spike_mask + self.voltage * ~self.spike_mask
        )

        if self.args.threshold_adaptation is not None and training:
            self.theta += self.args.threshold_adaptation.theta_spike * self.spike_mask


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

    def step(self, step_time: float, training: bool) -> None:
        for neurons in self.neurons:
            neurons.step(step_time, training)

        for synapses in self.synapses:
            synapses.step()


def build_network(args: NetworkArgs, ckpt_path: Path | None = None) -> Network:
    size = args.hidden_size

    pg_inp = PoissonGroup(
        size=args.input_size,
        rate=torch.zeros(args.input_size),
    )

    # TODO: @roman move neuron configs to network args
    ng_exc_config = NeuronConfig["EXC"]
    if ckpt_path is not None:
        # theta is stored in volts - TODO: @roman refactor brian2 script to avoid this
        theta = torch.tensor(1000 * np.load(ckpt_path / "theta.npy"), dtype=torch.float)
    else:
        assert ng_exc_config.threshold_adaptation is not None
        theta = ng_exc_config.threshold_adaptation.theta_init * torch.ones(size)

    ng_exc = NeuronGroup(
        ng_exc_config,
        size=size,
        theta=theta,
    )

    ng_inh = NeuronGroup(
        NeuronConfig["INH"],
        size=size,
        theta=torch.zeros(size),
    )

    if ckpt_path is not None:
        weight = (
            torch.tensor(np.load(ckpt_path / "weights.npy"), dtype=torch.float)
            .reshape(args.input_size, size)
            .T
        )
    else:
        if args.weight.weight_inp_exc_init_proba is not None:
            weight = (
                torch.rand(size, args.input_size)
                < args.weight.weight_inp_exc_init_proba
            ).float()
        elif args.weight.weight_inp_exc_scale is not None:
            weight = args.weight.weight_inp_exc_scale * torch.rand(
                size, args.input_size
            )
        else:
            raise ValueError(f"Incorrect weight initalisation {args.weight}")

    syns_inp_exc = SynapseGroup(
        pre=pg_inp,
        post=ng_exc,
        exc_weight=weight,
        inh_weight=torch.zeros(size, args.input_size),
    )
    syns_exc_inh = SynapseGroup(
        pre=ng_exc,
        post=ng_inh,
        exc_weight=args.weight.weight_exc_inh * torch.eye(size),
        inh_weight=torch.zeros(size, size),
    )
    syns_inh_exc = SynapseGroup(
        pre=ng_inh,
        post=ng_exc,
        exc_weight=torch.zeros(size, size),
        inh_weight=args.weight.weight_inh_exc
        * (torch.ones(size, size) - torch.eye(size)),
    )

    return Network(
        args=args,
        neurons=[pg_inp, ng_exc, ng_inh],
        synapses=[syns_inp_exc, syns_exc_inh, syns_inh_exc],
    )
