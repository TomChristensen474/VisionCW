from dataclasses import dataclass
from task3Tom import TemplateImageKeypointMatch

import numpy as np
import numpy.typing as npt
import random


@dataclass
class Point:
    x: int
    y: int


@dataclass
class Homography:
    matrix: npt.NDArray[np.float64]  # 3x3 matrix


class Ransac:

    def __init__(self, distance_threshold: float = 10):
        self.distance_threshold = distance_threshold
        self.sample_points_num = 4  # TODO try picking 4 from a larger sample size with best SSD

    # four point algorithm from lectures
    def four_point_algorithm(self, p: np.ndarray, q: np.ndarray) -> Homography:
        """
        Estimate a 2D transformation using the four point algorithm.
        Args:
        p (ndarray): 3x4 array of homogeneous coordinates of points in the first image.
        q (ndarray): 3x4 array of homogeneous coordinates of corresponding points in the second image.
        Returns: H (ndarray): 3x3 transformation matrix that maps points in the first image to their corresponding points in the second image.
        """
        A = np.zeros((8, 9))
        for i in range(4):
            A[2 * i, 0:3] = p[:, i]
            A[2 * i, 6:9] = -q[0, i] * p[:, i]
            A[2 * i + 1, 3:6] = p[:, i]
            A[2 * i + 1, 6:9] = -q[1, i] * p[:, i]

        # Solve the homogeneous linear system using SVD
        U, D, Vt = np.linalg.svd(A)
        H = Vt[-1, :].reshape(3, 3)

        # Normalize the solution to ensure H[2, 2] = 1
        H = H / H[2, 2]
        H = Homography(matrix=H)
        return H

    """
    points:
    [[x1,y1],
    [x2,y2],
    [x3,y3],
    [x4,y4]]
    """

    def apply_homography_transform(self, H: Homography, points) -> list[Point]:
        """
        points_homogeneous
        [[x1 x2 x3 x4],
        [y1,y2,y3,y4],
        [1,1,1,1]]
        """
        points_homogeneous = np.vstack([points.T, np.ones((1, points.shape[0]))])

        transformed_points = H.matrix @ points_homogeneous

        transformed_points /= transformed_points[2, :]

        """
        points_cartesian_form
        [[x1,y1],
        [x2,y2],
        [x3,y3],
        [x4,y4]]
        """
        points_cartesian_form = transformed_points[:2, :].T

        transformed_points = []

        for point in points_cartesian_form:
            point = Point(int(point[0]), int(point[1]))
            transformed_points.append(point)

        return transformed_points

    # takes the points not fitted to line for calculation
    def calculate_homography_outliers(
        self, points: list[TemplateImageKeypointMatch]
    ) -> tuple[list[Point], Homography]:
        outliers = []

        def calc_homography(points: list[TemplateImageKeypointMatch]) -> Homography:
            p = np.zeros((3, 4))
            q = np.zeros((3, 4))

            for i, point in enumerate(points):  # should be 4 points in list
                p[0, i] = point.template_point.x
                p[1, i] = point.template_point.y
                p[2, i] = 1

                q[0, i] = point.image_point.x
                q[1, i] = point.image_point.y
                q[2, i] = 1

            return self.four_point_algorithm(p, q)

        def apply_homography(points: list[TemplateImageKeypointMatch], homography: Homography) -> list[Point]:
            return self.apply_homography_transform(homography, points)

        homography = calc_homography(points)
        transformed_points = apply_homography(points, homography)

        def calculate_distance_between_points(point1: Point, point2: Point) -> float:
            return np.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

        for index, point in enumerate(transformed_points):
            point2 = points[index].image_point
            distance = calculate_distance_between_points(point, point2)
            if distance > self.distance_threshold:
                outliers.append(point)

        return outliers, homography

    # random sampling of points for line fitting and inlier outlier calculation
    def sample_points(self, points: list[TemplateImageKeypointMatch]) -> list[TemplateImageKeypointMatch]:
        # chosen_point_indices = np.random.choice(len(points), self.sample_points_num, replace=False)
        return random.sample(points, self.sample_points_num)

    def run_ransac(self, points: list[TemplateImageKeypointMatch], iterations=50) -> tuple[list[Point], Homography]:
        print("Running RANSAC")
        best_outlier_count = None  # less outliers is better
        best_points = []
        best_line = None

        for i in range(iterations):
            sampled_points = self.sample_points(points)
            outliers, homography = self.calculate_homography_outliers(sampled_points)

            if best_outlier_count is None or len(outliers) < best_outlier_count:
                best_outlier_count = len(outliers)
                best_homography = homography

        if not best_homography:  # if empty arry or best line is none
            raise ValueError("no homography found")

        # re fit and get parameters based on best line
        final_m, final_c = self.fit_line(selected_points=best_points)
        best_line = LineEquation(m=final_m, c=final_c)

        return best_points, best_line


if __name__ == "__main__":
    rsc = Ransac()
    points = [Point(1, 6), Point(3, 7)]
    new_points = rsc.run_ransac(points=points)
