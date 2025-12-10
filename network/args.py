from dataclasses import dataclass


@dataclass
class NeuronArgs:
    # resting potential (mV)
    v_rest: float

    # reset potential (mV)
    v_reset: float

    # membrane potential relaxation time (ms)
    tau_membrane: float

    # excitatory/inhibitory reversal potential (mV)
    v_exc_base: float
    v_inh_base: float

    # excitatory/inhibitory synaptic conductance decay time (ms)
    tau_g_exc: float
    tau_g_inh: float

    # firing threshold shift (mV)
    theta_shift: float

    # firing threshold decay time (ms)
    tau_theta: float

    # firing threshold increase when spiking (mV)
    theta_spike: float

    # potential/timer refractory time (ms)
    v_refractory: float
    timer_refractory: float | None

    @property
    def spike_refractory_threshold(self) -> float:
        return max(self.v_refractory, self.timer_refractory or 0.0)


@dataclass
class NetworkArgs:
    # number of input neurons and image size
    input_size: int

    # number of exc/inh neurons
    hidden_size: int

    # simulation step time (ms)
    step_time: float

    # strength of exc -> inh and inh -> exc synaptic connection
    weight_exc_inh: float
    weight_inh_exc: float


@dataclass
class EvalArgs:
    # MNIST test samples
    num_classes: int
    num_samples: int

    # stimulation time (ms)
    stimulation_time: float

    # rest after stimulation time (ms)
    rest_time: float

    # spike repeat threshold
    # if there are less spikes during stimulation, repeat with higher intenstity
    spike_threshold: int

    # intensity
    starting_intensity: float
    intensity_increase: float
