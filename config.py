from args import CalciumArgs, SimulationArgs, SynapticArgs

FigConfig: dict[str, SimulationArgs] = {
    "DP": SimulationArgs(
        calcium=CalciumArgs(
            tau_ca=20,
            c_pre=1,
            c_post=2,
            D=13.7,
        ),
        synapse=SynapticArgs(
            tau=150_000,
            rho_star=0.5,
            gamma_d=200,
            gamma_p=321.808,
            theta_d=1,
            theta_p=1.3,
            sigma=2.8284,
            up_down_strength_ratio=5,
        ),
        simulation_time=60_000,
        step_time=0.1,
        spike_rate=0.001,
        pre_post_delay=0,
        down_init_probability=0.5,
    )
}
