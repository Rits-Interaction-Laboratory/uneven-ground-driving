import time


class Position:
    def __init__(self, x, y, z):
        self.x: float = x
        self.y: float = y
        self.z: float = z
        self.timestamp: float = time.time()

    def to_json(self) -> dict:
        return {
            'timestamp': self.timestamp,
            'x': self.x,
            'y': self.y,
            'z': self.z,
        }


class PositionRepository:
    def __init__(self):
        self.__positions: list[Position] = []

    def select_all(self) -> list[Position]:
        return self.__positions

    def select_latest_position(self) -> Position:
        return self.__positions[-1]

    def select_oldest_position(self) -> Position:
        return self.__positions[0]

    def insert(self, position: Position):
        self.__positions.append(position)

    def delete_all(self):
        self.__positions = []
