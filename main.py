from system import build_3_bus_system, PowerSystem

import pdb


def build_ps():

    sys = PowerSystem(name="3 Bus", base_voltage_kv=100, base_power_mva=100)

    bus_k1 = sys.create_bus(name="K1", type="pv", v=1.0, theta=0.0, bsh=0.057)
    bus_k2 = sys.create_bus(name="K2", type="slack", v=1.02, theta=0.0, bsh=0.1)
    bus_k3 = sys.create_bus(name="K3", type="pq", v=1.035, theta=0.0, bsh=0.182)

    l1 = sys.create_load(name="L1", p_c=1.80, q_c=0.0, bus=bus_k1.name)
    l2 = sys.create_load(name="L2", p_c=2.65, q_c=0.0, bus=bus_k2.name)
    l3 = sys.create_load(name="L3", p_c=1.08, q_c=0.0, bus=bus_k3.name)

    g1 = sys.create_generator(
        name="G1",
        type="thermo",
        p_g_max=1.0,
        p_g_min=0.0,
        q_g_min=0.0,
        q_g_max=0.0,
        bus=bus_k2.name,
        a=100.0,
        b=4000.0,
        c=0.0,
    )
    g2 = sys.create_generator(
        name="G2",
        type="thermo",
        p_g_max=1.08,
        p_g_min=0.0,
        q_g_min=0.0,
        q_g_max=0.0,
        bus=bus_k2.name,
        a=12500.0,
        b=2000.0,
        c=0.0,
    )
    g3 = sys.create_generator(
        name="G3",
        type="thermo",
        p_g_max=1.5,
        p_g_min=0.0,
        q_g_min=0.0,
        q_g_max=0.0,
        bus=bus_k2.name,
        a=200.0,
        b=2000.0,
        c=0.0,
    )

    g4 = sys.create_generator(
        name="G4",
        type="hydro",
        p_g_max=1.05,
        p_g_min=0.0,
        q_g_min=-0.60,
        q_g_max=0.60,
        bus=bus_k3.name,
    )
    g5 = sys.create_generator(
        name="G5",
        type="hydro",
        p_g_max=1.80,
        p_g_min=0.0,
        q_g_min=-0.90,
        q_g_max=1.68,
        bus=bus_k3.name,
    )
    g4 = sys.create_generator(
        name="G4",
        type="hydro",
        p_g_max=1.64,
        p_g_min=0.0,
        q_g_min=-0.08,
        q_g_max=0.24,
        bus=bus_k3.name,
    )
    sys.initialize()
    

    return sys


if __name__ == "__main__":

    # t = build_3_bus_system()
    t = build_ps()

    pdb.set_trace()
