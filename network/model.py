from dataclasses import dataclass

import torch


@dataclass
class NeuronArgs:
    # resting potential (mV)
    v_rest: float

    # membrane potential relaxation time (ms)
    tau_membrane: float

    # excitatory/inhibitory reversal potential (mV)
    v_exc_base: float
    v_inh_base: float

    # excitatory/inhibitory synaptic conductance decay time (ms)
    tau_g_exc: float
    tau_g_inh: float

    # firing threshold initialisation (mV)
    theta_init: float

    # firing threshold shift (mV)
    theta_shift: float

    # firing threshold decay time (ms)
    tau_theta: float

    # firing threshold increase when spiking (mV)
    theta_spike: float

    # potential/timer refractory time (ms)
    v_refractory: float
    timer_refractory: float


class NeuronGroup:
    def __init__(self, args: NeuronArgs, size: int, step_time: float) -> None:
        self.args = args
        self.size = size
        self.step_time = step_time

        self.voltage = args.v_rest * torch.ones(size)
        self.g_exc = torch.zeros(size)
        self.g_inh = torch.zeros(size)
        self.theta = args.theta_init * torch.ones(size)
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
        spike_mask = (self.voltage > self.theta + self.args.theta_shift) & (
            self.timer > max(self.args.v_refractory, self.args.timer_refractory)
        )
        self.voltage = self.args.v_rest * spike_mask + self.voltage * (1 - spike_mask)
        self.theta += self.args.theta_spike * spike_mask

class Synapses:
    def __init__(self, source: NeuronGroup, target: NeuronGroup, weights: torch.Tensor) -> None:
        self.source = source
        self.target = target
        self.weights = weights