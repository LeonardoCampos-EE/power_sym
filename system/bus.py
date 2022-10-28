import numpy as np
from typing import List, Union


class Bus:
    def __init__(
        self,
        name: str,
        index: int,
        type: str,
        v: float = 1.0,
        theta: float = 0.0,
        bsh: float = 0.0,
    ) -> None:

        """
        The Bus shall follow generator's sign convention for power:

        Generation -> Power > 0
        Consumption -> Power < 0

        """

        # Bus name
        self.name: str = name
        # Bus index in the system
        self.index: int = index
        # Bus type: slack, PQ or PV
        self.type: str = type
        # Voltage magnitude (pu)
        self.v: float = v
        # Voltage angle (rad)
        self.theta: float = theta
        # Shunt susceptance (pu)
        self.bsh: float = bsh

        # Active power generated
        self.p_g: float = 0.0
        # Active power consumed
        self.p_c: float = 0.0
        # Reactive power generated
        self.q_g: float = 0.0
        # Reactive power consumed
        self.q_c: float = 0.0
        # Indexes of generators connected to this bus
        self.generators: List[int] = []
        # Indexes of loads connected to this bus
        self.loads: List[int] = []
        # Indexes of the buses that this bus is connected to
        self.connections: List[int] = []

    def add_generator(
        self, generator_index: int, generator_p_g: float, generator_q_g: float
    ) -> None:

        self.generators.append(generator_index)

        # Keep the generation sign convention
        if generator_p_g > 0:
            self.p_g += generator_p_g
        else:
            self.p_c += abs(generator_p_g)

        if generator_q_g > 0:
            self.q_g += generator_q_g
        else:
            self.q_c += abs(generator_q_g)

        return

    def add_load(self, load_index: int, load_p_c: float, load_q_c: float) -> None:

        self.loads.append(load_index)

        # Keep the generation sign convention
        if load_p_c > 0:
            self.p_c += load_p_c
        else:
            self.p_g += abs(load_p_c)

        if load_q_c > 0:
            self.q_c += load_q_c
        else:
            self.q_g += abs(load_q_c)

        return

    def add_connection(self, connected_bus: int) -> None:
        self.connections.append(connected_bus)

        return

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "type": self.type,
            "v": self.v,
            "theta": self.theta,
            "bsh": self.bsh,
            "p_g": self.p_g,
            "p_c": self.p_c,
            "q_g": self.q_g,
            "q_c": self.q_c,
        }
