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
            tau=3_000,
            rho_star=0.5,
            gamma_d=4,
            gamma_p=6.4316,
            theta_d=1,
            theta_p=1.3,
            sigma=0.4,
        ),
        simulation_time=60_000,
        step_time=1,
        spike_rate=0.001,
        pre_post_delay=0,
        rho_init=0.5,
    )
}
