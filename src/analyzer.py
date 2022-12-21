import json

import numpy as np


class Result:
    def __init__(self, src: dict):
        self.image_filename: str = src['image_filename']
        self.odometries = [self.Odometry(odometry_src) for odometry_src in src['odometries']]

    def get_movement_amount(self) -> tuple[float, float]:
        """
        移動量を取得
        """

        return (
            self.odometries[-1].position.x - self.odometries[0].position.x,
            self.odometries[-1].position.y - self.odometries[0].position.y,
        )

    class Odometry:
        def __init__(self, src: dict):
            self.position = self.Position(src['position'])
            self.orientation = self.Orientation(src['orientation'])

        class Position:
            def __init__(self, src: dict):
                self.x: float = src['x']
                self.y: float = src['y']
                self.z: float = src['z']

        class Orientation:
            def __init__(self, src: dict):
                self.x: float = src['x']
                self.y: float = src['y']
                self.z: float = src['z']
                self.w: float = src['w']


results: list[Result] = []
with open('../.ros/uneven-ground-driving-result/result.jsonl', 'r') as f:
    for line in f.readlines():
        results.append(Result(json.loads(line)))

movement_amounts: list[tuple[float, float]] = [result.get_movement_amount() for result in results]

xの平均 = np.mean([movement_amount[0] for movement_amount in movement_amounts])
yの平均 = np.mean([movement_amount[1] for movement_amount in movement_amounts])
print(f'件数: {len(movement_amounts)}')
print(f'xの平均: {xの平均}')
print(f'yの平均: {yの平均}')
print(np.cov(movement_amounts, rowvar=0))
