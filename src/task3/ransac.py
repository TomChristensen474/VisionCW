from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

@dataclass
class Point:
    x: int
    y: int

class Ransac:

    def __init__(self, distance_threshold: float = 0.5, sample_points_num: int = 100):
        self.distance_threshold = distance_threshold
        self.sample_points_num = 


    # random sampling of points
    def sample_points(self, points: List[Point]):
        chosen_point_indices = np.random.choice(len(points), self.sample_points_num, replace=False)
        sampled_points = []
        for i, point in enumerate(points):
            if i in chosen_point_indices:
                sampled_points.append(point)

        return sampled_points

    # fit points to linear line using least squares
    def fit_line(self, points: List[Point]) -> Tuple[float,float]:
        x_coords = []
        y_coords = []
        for point in points:
            x_coords.append(point.x)
            y_coords.append(point.y)

        coefficients = np.polyfit(np.asarray(x_coords), np.asarray(y_coords), 1)
        m, c = coefficients
        return m,c

    def run_ransac(self, points):
        sampled_points = self.sample_points(points=points)
        m,c = self.fit_line(points=sampled_points)

if __name__ == '__main__':
    rsc = Ransac()
    sample_points = [Point(1,6), Point(3,7)]
    rsc.run_ransac(points=sample_points)
