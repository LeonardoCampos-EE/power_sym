import numpy as np
import pandas as pd
from typing import List, Type, Union

from .bus import Bus
from .line import Line
from .generator import Generator
from .load import Load


class PowerSystem:
    def __init__(
        self, name: str, base_voltage_kv: float, base_power_mva: float
    ) -> None:
        self.name = name
        self.base_voltage_kv = base_voltage_kv
        self.base_power_mva = base_power_mva
        self.base_impedance = ((self.base_voltage_kv * 1000) ** 2) / (
            self.base_power_mva * 1e6
        )

        self.buses: List[Bus] = []
        self.lines: List[Line] = []
        self.loads: List[Load] = []
        self.generators: List[Generator] = []

        return

    def create_bus(
        self,
        name: str,
        type: str,
        v: Union[float, None] = None,
        theta: Union[float, None] = None,
        bsh: Union[float, None] = None,
    ) -> Bus:

        bus = Bus(
            name=name,
            index=len(self.buses),
            type=type,
            v=v if v is not None else np.nan,
            theta=theta if theta is not None else np.nan,
            bsh=bsh if bsh is not None else np.nan,
        )

        self.buses.append(bus)

        return bus

    def _find_bus_by_name(self, name: str) -> Union[Bus, None]:

        bus = list(filter(lambda x: x.name == name, self.buses))

        if len(bus) == 0:
            return None
        else:
            return bus[0]

    def _add_element_to_bus(self, bus: Bus, element: Union[Generator, Load]) -> None:

        if isinstance(element, Generator):
            bus.add_generator(
                generator_index=len(self.generators),
                generator_p_g=element.p_g,
                generator_q_g=element.q_g,
            )
        elif isinstance(element, Load):
            bus.add_load(
                load_index=len(self.loads), load_p_c=element.p_c, load_q_c=element.q_c
            )
        else:
            raise TypeError("Element is neither a Generator or a Load")

        return

    def _add_connection_between_buses(self, start: Bus, end: Bus) -> None:

        start.add_connection(end.index)
        end.add_connection(start.index)

        return

    def create_generator(
        self,
        name: str,
        p_g_max: float,
        p_g_min: float,
        q_g_max: float,
        q_g_min: float,
        type: str = "thermo",
        bus: Union[str, None] = None,
        is_fixed_power: bool = False,
        p_g: Union[float, None] = None,
        q_g: Union[float, None] = None,
        a: Union[float, None] = None,
        b: Union[float, None] = None,
        c: Union[float, None] = None,
        power_goal: Union[float, None] = None,
    ) -> Generator:

        """
        If the generator hasn't a fixed power output, its power is initialized as the
        mean between the minimum and maximum generation.
        """

        # Find bus by name
        bus_ = self._find_bus_by_name(bus)

        if bus_ is None:
            raise ValueError(
                "Bus does not exist in the system, make sure the name is correct"
            )

        generator = Generator(
            name=name,
            index=len(self.generators),
            p_g_max=p_g_max,
            p_g_min=p_g_min,
            q_g_max=q_g_max,
            q_g_min=q_g_min,
            type=type,
            bus=bus_.index,
            is_fixed_power=is_fixed_power,
            p_g=np.mean([p_g_max, p_g_min]) if not is_fixed_power else p_g,
            q_g=np.mean([q_g_max, q_g_min]) if not is_fixed_power else q_g,
            a=a,
            b=b,
            c=c,
            power_goal=None if type != "hydro" else power_goal,
        )

        try:
            self._add_element_to_bus(bus_, generator)
        except TypeError:
            raise TypeError("Incorrect element type, internal error")

        self.generators.append(generator)

        return generator

    def create_load(
        self, name: str, p_c: float, q_c: float, bus: Union[str, None] = None
    ) -> Load:

        """
        If the generator hasn't a fixed power output, its power is initialized as the
        mean between the minimum and maximum generation.
        """

        # Find bus by name
        bus_ = self._find_bus_by_name(bus)

        if bus_ is None:
            raise ValueError(
                "Bus does not exist in the system, make sure the name is correct"
            )

        load = Load(
            name=name,
            index=len(self.loads),
            p_c=p_c,
            q_c=q_c,
            bus=bus_.index,
        )

        try:
            self._add_element_to_bus(bus_, load)
        except TypeError:
            raise TypeError("Incorrect element type, internal error")

        self.loads.append(load)

        return load

    def create_line(self, start_bus: str, end_bus: str, r: float, x: float) -> Line:

        # Find bus by name
        start_bus_ = self._find_bus_by_name(start_bus)

        if start_bus_ is None:
            raise ValueError(
                "Start bus does not exist in the system, make sure the name is correct"
            )

        # Find bus by name
        end_bus_ = self._find_bus_by_name(end_bus)

        if end_bus_ is None:
            raise ValueError(
                "End bus does not exist in the system, make sure the name is correct"
            )

        line = Line(
            start=start_bus_.index, end=end_bus_.index, index=len(self.lines), r=r, x=x
        )

        self._add_connection_between_buses(start_bus_, end_bus_)

        self.lines.append(line)
        return

    def initialize(self):
        # Get the array of line resistances
        self.r = np.array([l.r for l in self.lines])

        # Get the array of line resistances
        self.x = np.array([l.x for l in self.lines])

        self.G = self.get_conductance_matrix()
        self.B = self.get_susceptance_matrix()

        self.initialize_dataframes()

    def initialize_dataframes(self) -> None:

        """
        This function creates dataframes for the buses, generators, loads and lines, similar to what is done
        in the PandaPower package
        """

        self.bus_info = pd.DataFrame([bus.to_dict() for bus in self.buses])
        self.gen_info = pd.DataFrame([gen.to_dict() for gen in self.generators])
        self.line_info = pd.DataFrame([line.to_dict() for line in self.lines])
        self.load_info = pd.DataFrame([load.to_dict() for load in self.loads])

        return

    def get_conductance_matrix(self) -> np.ndarray:
        # Get number of buses
        nb = len(self.buses)

        # Compute the conductance for each line
        conductances = self.r / ((self.r**2) + (self.x**2))

        # Initialize the conductance matrix
        conductance_matrix = np.zeros(shape=(nb, nb), dtype=np.float32)

        # Get the starting bus of all lines
        starts = [l.start for l in self.lines]

        # Get the ending bus of all lines
        endings = [l.end for l in self.lines]

        # Fill the conductance matrix with the correct conductances
        conductance_matrix[starts, endings] = conductances

        return conductance_matrix

    def get_susceptance_matrix(self) -> np.ndarray:
        # Get number of buses
        nb = len(self.buses)

        # Compute the susceptance for each line
        susceptances = -self.x / ((self.r**2) + (self.x**2))

        # Initialize the susceptance matrix
        susceptance_matrix = np.zeros(shape=(nb, nb), dtype=np.float32)

        # Get the starting bus of all lines
        starts = [l.start for l in self.lines]

        # Get the ending bus of all lines
        endings = [l.end for l in self.lines]

        # Fill the susceptance matrix with the correct susceptances
        susceptance_matrix[starts, endings] = susceptances

        return susceptance_matrix
