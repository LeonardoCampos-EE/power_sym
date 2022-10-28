import pandapower as pp
import numpy as np


class PowerSystem:
    def __init__(self, system: pp.pandapowerNet) -> None:
        self.system = system

        self._sort_system_indexes()
        self._get_number_of_system_elements()
        self._get_transmission_line_parameters()
        self._get_conductance_matrix()

        return

    def _sort_system_indexes(self) -> None:

        # Ordena os indíces dos elementos do sistema
        self.network.bus = self.network.bus.sort_index()
        self.network.res_bus = self.network.res_bus.sort_index()
        self.network.gen = self.network.gen.sort_index()
        self.network.line = self.network.line.sort_index()
        self.network.shunt = self.network.shunt.sort_index()
        self.network.trafo = self.network.trafo.sort_index()

        # Soluciona problema em que o PP inicia alguns taps negativos
        self.network.trafo.tap_pos = np.abs(self.network.trafo.tap_pos)

        return

    def _get_number_of_system_elements(self) -> None:

        # Determina o número de trafos com controle de tap
        self.network.trafo = self.network.trafo.sort_values("tap_pos")
        self.nt = self.network.trafo.tap_pos.count()

        # Determina o número de barras do sistema
        self.nb = self.network.bus.name.count()

        # Determina o número de susceptâncias shunt do sistema
        self.ns = self.network.shunt.in_service.count()

        # Determina o número de geradores do sistema
        self.ng = self.network.gen.in_service.count()

    def _get_transmission_line_parameters(self) -> None:

        # Inicializa o dicionário que conterá as linhas de transmissão
        lines = {}

        # Armazena os indices das barras de onde iniciam as linhas de transmissão
        lines["start"] = self.network.line.from_bus.to_numpy().astype(int)

        # Armazena os indices das barras onde terminam as linhas de transmissão
        lines["end"] = self.network.line.to_bus.to_numpy().astype(int)

        # Determina a tensão de base do sistema em kV
        v_network = self.network.bus.vn_kv.to_numpy()

        # Determina a impedância de base do sistema
        # Z = V²/100M
        z_base = ((v_network * 1000) ** 2) / 100e6

        # Determina as resistências do sistema em PU
        # r_pu = r_ohm / z_base
        lines["r_pu"] = np.zeros(shape=(self.network.line.index[-1] + 1,))
        for i in range(self.network.line.index[-1] + 1):
            lines["r_pu"][i] = (
                self.network.line.r_ohm_per_km.iloc[i] / z_base[lines["start"][i]]
            )

        # Determina as reatâncias do sistema em PU
        # x_pu = x_ohm / z_base
        lines["x_pu"] = np.zeros(shape=(self.network.line.index[-1] + 1,))
        for i in range(self.network.line.index[-1] + 1):
            lines["x_pu"][i] = (
                self.network.line.x_ohm_per_km.iloc[i] / z_base[lines["start"][i]]
            )

        self.lines = lines

        return

    def _get_conductance_matrix(self) -> None:

        conductances = np.array(
            [self.lines["r_pu"] / self.lines["r_pu"] ** 2 + self.lines["x_pu"] ** 2]
        )

        # Get the nodal conductance matrix, it's equivalent to the real part of the nodal admitance matrix
        self.conductance_matrix = np.zeros(shape=(self.nb, self.nb))
        self.conductance_matrix[self.lines["start"], self.lines["end"]] = conductances

        return

    def run_power_flow(self) -> None:

        pp.runpp(
            self.system,
            algorithm="fdbx",
            numba=True,
            init="results",
            enforce_q_lims=False,
            tolerance_mva=1e-5,
        )

        return