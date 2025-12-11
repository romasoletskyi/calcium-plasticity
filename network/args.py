from dataclasses import dataclass

from args import CalciumArgs, SynapticArgs


@dataclass
class ThresholdAdaptationArgs:
    # threshold decay time (ms)
    tau_theta: float

    # threshold increase when spiking (mV)
    theta_spike: float

    # threshold starting value when training (mV)
    theta_init: float


@dataclass
class NeuronArgs:
    # resting potential (mV)
    v_rest: float

    # reset potential after spike (mV)
    v_reset: float

    # threshold potential for spike (mV)
    v_threshold: float

    # membrane potential relaxation time (ms)
    tau_membrane: float

    # excitatory/inhibitory reversal potential (mV)
    v_exc_base: float
    v_inh_base: float

    # excitatory/inhibitory synaptic conductance decay time (ms)
    tau_g_exc: float
    tau_g_inh: float

    # Chapter 1.4.1 of Neuronal Dynamics (Gerstner et al.)
    threshold_adaptation: ThresholdAdaptationArgs | None

    # potential refractory time (ms)
    v_refractory: float

    # spiking refractory time (ms)
    timer_refractory: float | None

    @property
    def spike_refractory_threshold(self) -> float:
        return max(self.v_refractory, self.timer_refractory or 0.0)


@dataclass
class SynapseArgs:
    calcium: CalciumArgs
    synapse: SynapticArgs


@dataclass
class WeightArgs:
    # strength of exc -> inh and inh -> exc synaptic connection
    weight_exc_inh: float
    weight_inh_exc: float

    # initialise weights as ~binomial(proba)
    weight_inp_exc_init_proba: float | None

    # initialise weights as ~uniform[0, scale]
    weight_inp_exc_scale: float | None

    def __post_init__(self) -> None:
        assert (
            sum(
                [
                    self.weight_inp_exc_init_proba is not None,
                    self.weight_inp_exc_scale is not None,
                ]
            )
            == 1
        ), "can choose only one init method"


@dataclass
class NetworkArgs:
    # number of input neurons and image size
    input_size: int

    # number of exc/inh neurons
    hidden_size: int

    exc_neuron: NeuronArgs
    inh_neuron: NeuronArgs
    synapse: SynapseArgs | None

    weight: WeightArgs


@dataclass
class SampleArgs:
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


@dataclass
class EvalArgs:
    # MNIST test samples
    num_classes: int
    num_samples: int

    sample: SampleArgs
