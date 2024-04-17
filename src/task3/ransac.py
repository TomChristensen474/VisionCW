from dataclasses import dataclass
from task3 import TemplateImageKeypointMatch, apply_homography_transform, Point, Homography

import numpy as np
import random

@dataclass
class inlier:
    template_point: Point
    image_point: Point


class Ransac:

    def __init__(self, distance_threshold: float = 10):
        self.distance_threshold = distance_threshold
        self.sample_points_num = 4  # TODO try picking 4 from a larger sample size with best SSD

    # four point algorithm from lectures
    def four_point_algorithm(self, p: np.ndarray, q: np.ndarray, num_points=4) -> Homography:
        """
        Estimate a 2D transformation using the four point algorithm.
        Args:
        p (ndarray): 3x4 array of homogeneous coordinates of points in the first image.
        q (ndarray): 3x4 array of homogeneous coordinates of corresponding points in the second image.
        Returns: H (ndarray): 3x3 transformation matrix that maps points in the first image to their corresponding points in the second image.
        """
        A = np.zeros((2*num_points, 9))
        for i in range(num_points):
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

    # takes the points not fitted to line for calculation
    def calculate_homography_outliers(
        self, points: list[TemplateImageKeypointMatch]
    ) -> tuple[list[inlier], list[Point], Homography]:
        outliers = []
        inliers = []

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
            points_to_transform = np.array([[point.template_point.x, point.template_point.y] for point in points])
            
            return apply_homography_transform(homography, points_to_transform)


        sampled_points = self.sample_points(points)
        homography = calc_homography(sampled_points)
        transformed_points = apply_homography(points, homography)

        def calculate_distance_between_points(point1: Point, point2: Point) -> float:
            return np.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)
        
        for index, point in enumerate(transformed_points):
            point2 = points[index].image_point
            distance = calculate_distance_between_points(point, point2)

            if distance > self.distance_threshold:
                outliers.append(point)
            else:
                inliers.append(inlier(template_point=points[index].template_point, image_point=points[index].image_point))

        return inliers, outliers, homography

    # random sampling of points for line fitting and inlier outlier calculation
    def sample_points(self, points: list[TemplateImageKeypointMatch]) -> list[TemplateImageKeypointMatch]:
        # random_sample = random.sample(points, 30) # random choice
        # sorted_sample = sorted(random_sample, key=lambda x: x.match_ssd)
        # return sorted_sample[:4]

        return random.sample(points, self.sample_points_num) # random choice of 4 points
    
    def refine_homography(self, inliers: list[inlier]) -> Homography:
        p = np.zeros((3, len(inliers)))
        q = np.zeros((3, len(inliers)))

        for i, point in enumerate(inliers):
            p[0, i] = point.template_point.x
            p[1, i] = point.template_point.y
            p[2, i] = 1

            q[0, i] = point.image_point.x
            q[1, i] = point.image_point.y
            q[2, i] = 1

        return self.four_point_algorithm(p, q, len(inliers))


    def run_ransac(self, points: list[TemplateImageKeypointMatch], iterations=100) -> tuple[Homography, list[inlier], list[Point]]:
        # best_outlier_count = None  # fewer outliers better
        count_inliers = 0
        best_outliers = []
        best_inliers = []
        best_homography = None

        for i in range(iterations):
            # sampled_points = self.sample_points(points)
            inliers, outliers, homography = self.calculate_homography_outliers(points)

            if len(inliers) > count_inliers:
                count_inliers = len(inliers)
                best_inliers = inliers
                best_outliers = outliers
                best_homography = homography
        

        if not best_homography:  # if empty arry or best line is none
            raise ValueError("no homography found")

        return self.refine_homography(best_inliers), best_inliers, best_outliers
