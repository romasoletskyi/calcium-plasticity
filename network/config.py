from network.args import NeuronArgs

NeuronConfig: dict[str, NeuronArgs] = {
    "EXC": NeuronArgs(
        v_rest=-65,
        v_reset=-65,
        tau_membrane=100,
        v_exc_base=0,
        v_inh_base=-100,
        tau_g_exc=1,
        tau_g_inh=2,
        theta_shift=-72,
        tau_theta=1e7,
        theta_spike=0.05,
        v_refractory=5,
        timer_refractory=50,
    ),
    "INH": NeuronArgs(
        v_rest=-60,
        v_reset=-45,
        tau_membrane=10,
        v_exc_base=0,
        v_inh_base=-85,
        tau_g_exc=1,
        tau_g_inh=2,
        theta_shift=-40,
        tau_theta=float("inf"),
        theta_spike=0.0,
        v_refractory=2,
        timer_refractory=None,
    ),
}
