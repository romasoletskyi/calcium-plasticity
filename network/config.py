from config import CalciumConfig, SynapseConfig
from network.args import (
    NetworkArgs,
    NeuronArgs,
    SampleArgs,
    SynapseArgs,
    ThresholdAdaptationArgs,
    WeightArgs,
)

NeuronConfig: dict[str, NeuronArgs] = {
    "EXC": NeuronArgs(
        v_rest=-65,
        v_reset=-65,
        v_threshold=-72,
        tau_membrane=100,
        v_exc_base=0,
        v_inh_base=-100,
        tau_g_exc=1,
        tau_g_inh=2,
        threshold_adaptation=ThresholdAdaptationArgs(
            tau_theta=1e7,
            theta_spike=0.05,
            theta_init=20,
        ),
        v_refractory=5,
        timer_refractory=50,
    ),
    "INH": NeuronArgs(
        v_rest=-60,
        v_reset=-45,
        v_threshold=-40,
        tau_membrane=10,
        v_exc_base=0,
        v_inh_base=-85,
        tau_g_exc=1,
        tau_g_inh=2,
        threshold_adaptation=None,
        v_refractory=2,
        timer_refractory=None,
    ),
}

NetworkConfig: dict[str, NetworkArgs] = {
    "base": NetworkArgs(
        input_size=784,
        hidden_size=400,
        exc_neuron=NeuronConfig["EXC"],
        inh_neuron=NeuronConfig["INH"],
        synapse=None,
        weight=WeightArgs(
            weight_exc_inh=10.4,
            weight_inh_exc=17.0,
            weight_inp_exc_scale=0.3,
            weight_inp_exc_init_proba=None,
        ),
    ),
    "calcium": NetworkArgs(
        input_size=784,
        hidden_size=400,
        exc_neuron=NeuronConfig["EXC"],
        inh_neuron=NeuronConfig["INH"],
        synapse=SynapseArgs(
            calcium=CalciumConfig["DP"],
            synapse=SynapseConfig["DP"],
        ),
        weight=WeightArgs(
            weight_exc_inh=10.4,
            weight_inh_exc=17.0,
            weight_inp_exc_scale=None,
            weight_inp_exc_init_proba=0.1,
        ),
    ),
}

SampleConfig: dict[str, SampleArgs] = {
    "base": SampleArgs(
        stimulation_time=350,
        rest_time=150,
        spike_threshold=5,
        starting_intensity=2,
        intensity_increase=1,
    ),
}
