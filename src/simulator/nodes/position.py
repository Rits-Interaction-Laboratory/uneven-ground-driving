class Position:
    def __init__(self, x, y, z):
        self.x: float = x
        self.y: float = y
        self.z: float = z

    def __str__(self):
        return "Position: {x: %5.3f, y: %5.3f, z: %5.3f}" % (self.x, self.y, self.z)


class PositionRepository:
    def __init__(self):
        self.__positions: list[Position] = []

    def insert(self, position: Position):
        self.__positions.append(position)

    def select_latest_position(self) -> Position:
        return self.__positions[-1]

    def select_oldest_position(self) -> Position:
        return self.__positions[0]
