import pandapower as pp

import warnings

warnings.filterwarnings("ignore")


def build_3_bus_system(
    base_voltage_kv: float = 110.0, base_power_mva: float = 100.0
) -> pp.pandapowerNet:

    # Inicializar a rede
    system = pp.create_empty_network(name="Sistema 2 Geradores", sn_mva=base_power_mva)

    # Determina a impedância de base do sistema
    # Z = V²/100M
    z_base = ((base_voltage_kv * 1000) ** 2) / (base_power_mva * 1e6)

    # Adicionar barra PV de Geração
    bus_1 = pp.create_bus(
        system,
        vn_kv=1.0 * base_voltage_kv,
        name="K1 - PV",
        index=0,
        type="n"
    )

    # Adicionar barra Slack
    bus_2 = pp.create_bus(
        system,
        vn_kv=1.02 * base_voltage_kv,
        name="K2 - Slack",
        index=1,
        type="n"
    )
    # # Adicionar Grid externo - é modelado no PP como slack
    grid = pp.create_ext_grid(
        system,
        bus=bus_2,
        vm_pu=1.02,
        va_degree=0.0,
        name="Slack",
    )

    # Adicionar barra PQ de carga
    bus_3 = pp.create_bus(
        system,
        vn_kv=1.035 * base_voltage_kv,
        name="K3 - PQ",
        index=2,
        type="n"
    )

    # Adicionar geradores termoelétricos
    pp.create_gen(
        system,
        bus=bus_2,
        vm_pu=1.02,
        p_mw=1.0 * base_power_mva,
        name="G1 - Termo",
        max_p_mw=1.0 * base_power_mva,
        min_p_mw=0.0,
    )
    pp.create_gen(
        system,
        bus=bus_2,
        vm_pu=1.02,
        p_mw=1.08 * base_power_mva,
        name="G2 - Termo",
        max_p_mw=1.08 * base_power_mva,
        min_p_mw=0.0,
    )
    pp.create_gen(
        system,
        bus=bus_2,
        vm_pu=1.02,
        p_mw=1.5 * base_power_mva,
        name="G3 - Termo",
        max_p_mw=1.5 * base_power_mva,
        min_p_mw=0.0,
    )

    # Adicionar geradores hidroelétricos
    pp.create_gen(
        system,
        bus=bus_3,
        vm_pu=1.035,
        p_mw=1.05 * base_power_mva,
        name="G4 - Hidro",
        max_p_mw=1.05 * base_power_mva,
        min_p_mw=0.0,
        max_q_mvar=0.6 * base_power_mva,
        min_q_mvar=-0.6 * base_power_mva,
    )
    pp.create_gen(
        system,
        bus=bus_3,
        vm_pu=1.035,
        p_mw=1.80 * base_power_mva,
        name="G5 - Hidro",
        max_p_mw=1.80 * base_power_mva,
        min_p_mw=0.0,
        max_q_mvar=1.68 * base_power_mva,
        min_q_mvar=-0.90 * base_power_mva,
    )
    pp.create_gen(
        system,
        bus=bus_3,
        vm_pu=1.035,
        p_mw=1.64 * base_power_mva,
        name="G6 - Hidro",
        max_p_mw=1.64 * base_power_mva,
        min_p_mw=0.0,
        max_q_mvar=0.24 * base_power_mva,
        min_q_mvar=-0.08 * base_power_mva,
    )

    """
    O PandaPower aceita os elementos das linhas apenas em Ohms, então
    é necessário converter os valores de PU para ohm

    r_ohm = r_pu * z_base
    x_ohm = x_pu * z_base
    """
    # Adicionar as linhas de transmissão
    pp.create_line_from_parameters(
        system,
        from_bus=bus_1,
        to_bus=bus_3,
        length_km=1.0,
        r_ohm_per_km=0.055 * z_base,
        x_ohm_per_km=0.211 * z_base,
        c_nf_per_km=11.0,
        max_i_ka=0.96,
        type="ol",
    )
    pp.create_line_from_parameters(
        system,
        from_bus=bus_1,
        to_bus=bus_2,
        length_km=1.0,
        r_ohm_per_km=0.006 * z_base,
        x_ohm_per_km=0.048 * z_base,
        c_nf_per_km=11.0,
        max_i_ka=0.96,
        type="ol",
    )
    pp.create_line_from_parameters(
        system,
        from_bus=bus_2,
        to_bus=bus_3,
        length_km=1.0,
        r_ohm_per_km=0.011 * z_base,
        x_ohm_per_km=0.087 * z_base,
        c_nf_per_km=11.0,
        max_i_ka=0.96,
        type="ol",
    )

    # Adicionar os shunts
    pp.create_shunt(
        system,
        bus_1,
        q_mvar=0.057 * 100,
    )
    pp.create_shunt(
        system,
        bus_2,
        q_mvar=0.1 * 100,
    )
    pp.create_shunt(
        system,
        bus_3,
        q_mvar=0.182 * 100,
    )

    # Adicionar as cargas
    pp.create_load(system, bus_1, p_mw=1.80 * base_power_mva * 1.255)
    pp.create_load(system, bus_2, p_mw=2.65 * base_power_mva * 1.255)
    pp.create_load(system, bus_3, p_mw=1.08 * base_power_mva * 1.255)

    # Executar o fluxo de potência para inicializar as variáveis do sistema
    pp.runpp(
        system,
        algorithm="nr",
        numba=True,
        enforce_q_lims=True,
        tolerance_mva=1e-6,
    )

    return system
