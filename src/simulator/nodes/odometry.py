import time


class OdometryModel:
    """
    ロボット位置
    """

    def __init__(self, position_x: float, position_y: float, position_z: float, orientation_x: float,
                 orientation_y: float, orientation_z: float, orientation_w: float):
        self.position = self.Position(position_x, position_y, position_z)
        self.orientation = self.Orientation(orientation_x, orientation_y, orientation_z, orientation_w)
        self.timestamp: float = time.time()

    def to_json(self) -> dict:
        return {
            'timestamp': self.timestamp,
            'position': self.position.to_json(),
            'orientation': self.orientation.to_json(),
        }

    class Position:
        def __init__(self, x: float, y: float, z: float):
            self.x: float = x
            self.y: float = y
            self.z: float = z

        def to_json(self) -> dict:
            return {
                'x': self.x,
                'y': self.y,
                'z': self.z,
            }

    class Orientation:
        def __init__(self, x: float, y: float, z: float, w: float):
            self.x: float = x
            self.y: float = y
            self.z: float = z
            self.w: float = w

        def to_json(self) -> dict:
            return {
                'x': self.x,
                'y': self.y,
                'z': self.z,
                'w': self.w,
            }


class OdometryRepository:
    """
    In Memory Odometry Repository
    """

    def __init__(self):
        self.__odometries: list[OdometryModel] = []

    def select_all(self) -> list[OdometryModel]:
        return self.__odometries

    def select_latest_odometry(self) -> OdometryModel:
        return self.__odometries[-1]

    def select_oldest_odometry(self) -> OdometryModel:
        return self.__odometries[0]

    def insert(self, odometry: OdometryModel):
        self.__odometries.append(odometry)

    def delete_all(self):
        self.__odometries = []
