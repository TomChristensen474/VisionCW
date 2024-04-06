from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

@dataclass
class LineEquation:
    m: float  # gradient
    c: float  # y int

@dataclass
class Point:
    x: int
    y: int

class Ransac:

    def __init__(self, distance_threshold: float = 0.5, sample_points_num: int = 100):
        self.distance_threshold = distance_threshold
        self.sample_points_num = sample_points_num

    # takes the points not fitted to line for calculation
    def calculate_inliers_and_outliers(self, m_term:float, c_term:float, unselected_points: List[Point]) -> Tuple[int,int]:
        inliers_count = 0
        outliers_count = 0

        # d = |ax1 + by1 + c| / (a^2+b^2)^1/2
        # where p(x1,x2) and ax + by + c = 0, 
        def calculate_distance_from_line(x1, y1, a, c):
            b = -1  # from mx - y + c = 0
            return abs( (a * x1) + (b * y1) + c) / np.sqrt(a**2 + b**2)

        for point in unselected_points:

            distance = calculate_distance_from_line(x1=point.x, y1=point.y, a=m_term, c=c_term) 
            if distance <= self.distance_threshold:
                inliers_count += 1
            else:
                outliers_count += 1
        
        return inliers_count, outliers_count


    # random sampling of points for line fitting and inlier outlier calculation
    def sample_points(self, points: List[Point]) -> Tuple[List[Point], List[Point]]:
        chosen_point_indices = np.random.choice(len(points), self.sample_points_num, replace=False)
        sampled_points = []
        unselected_points = []
        for i, point in enumerate(points):
            if i in chosen_point_indices:
                sampled_points.append(point)
            else:
                unselected_points.append(point)

        return sampled_points, unselected_points

    # fit points to linear line using least squares
    def fit_line(self, selected_points: List[Point]) -> Tuple[float,float]:
        x_coords = []
        y_coords = []
        for point in selected_points:
            x_coords.append(point.x)
            y_coords.append(point.y)

        coefficients = np.polyfit(np.asarray(x_coords), np.asarray(y_coords), 1)
        m, c = coefficients
        return m,c

    def run_ransac(self, points: List[Point]):

        selected_points, unselected_points = self.sample_points(points=points)
        m,c = self.fit_line(selected_points=selected_points)
        inliers, outliers = self.calculate_inliers_and_outliers(m_term=m, c_term=c, unselected_points=unselected_points)
        line = LineEquation(m=m,c=c)

if __name__ == '__main__':
    rsc = Ransac()
    points = [Point(1,6), Point(3,7)]
    rsc.run_ransac(points=points)
