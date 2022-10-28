class Line:
    def __init__(self, start: int, end: int, index: int, r: float, x: float) -> None:

        # Index of the starting bus
        self.start: int = start
        # Index of the ending bus
        self.end: int = end
        # Index of this line in the system
        self.index: int = index
        # Resistance in pu
        self.r: float = r
        # Reactance in pu
        self.x: float = x

        # Conductance
        self.g = self.r / (self.r**2 + self.x**2)

        # Suceptance
        self.b = -self.x / (self.r**2 + self.x**2)

    def to_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "r": self.r,
            "x": self.x,
            "g": self.g,
            "b": self.b,
        }
