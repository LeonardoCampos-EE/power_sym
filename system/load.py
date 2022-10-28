from typing import Union


class Load:
    def __init__(
        self,
        name: str,
        index: int,
        p_c: float,
        q_c: float,
        bus: Union[str, None] = None,
    ) -> None:

        # Load name
        self.name: str = name
        # Load index in the system
        self.index: int = index
        # Active power consumed
        self.p_c: float = p_c
        # Reactive power consumed
        self.q_c: float = q_c

        # Bus the load is connected to
        self.bus: Union[int, None] = bus

    def to_dict(self) -> dict:

        return {
            "name": self.name,
            "p_c": self.p_c,
            "q_c": self.q_c,
            "bus": self.bus,
        }
