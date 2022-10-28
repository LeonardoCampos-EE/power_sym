from typing import Union


class Generator:
    def __init__(
        self,
        name: str,
        index: int,
        p_g: float,
        p_g_max: float,
        p_g_min: float,
        q_g: float,
        q_g_max: float,
        q_g_min: float,
        type: str = "thermo",
        is_fixed_power: bool = True,
        a: Union[float, None] = None,
        b: Union[float, None] = None,
        c: Union[float, None] = None,
        power_goal: Union[float, None] = None,
        bus: Union[str, None] = None,
    ) -> None:

        # Generator name
        self.name: str = name
        # Index of the bus this generator is connected to
        self.index: int = index
        # Active power generated
        self.p_g: float = p_g
        self.p_g_max: float = p_g_max
        self.p_g_min: float = p_g_min
        # Reactive power generated
        self.q_g: float = q_g
        self.q_g_max: float = q_g_max
        self.q_g_min: float = q_g_min

        # Can be thermo, hydro, wind or pv
        self.type: str = type

        # Consider the power generation as fixed (for power flow)
        self.is_fixed_power: bool = is_fixed_power

        # Quadratic cost params
        self.a: Union[float, None] = a
        self.b: Union[float, None] = b
        self.c: Union[float, None] = c

        # Power goal for hydro generators
        self.power_goal: Union[float, None] = power_goal

        # Bus the generator is connected to
        self.bus: Union[int, None] = bus

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "type": self.type,
            "p_g": self.p_g,
            "p_g_min": self.p_g_min,
            "p_g_max": self.p_g_max,
            "q_g": self.q_g,
            "q_g_min": self.q_g_min,
            "q_g_max": self.q_g_max,
            "fixed_power": self.is_fixed_power,
            "power_goal": self.power_goal,
            "a": self.a,
            "b": self.b,
            "c": self.c,
            "bus": self.bus,
        }
