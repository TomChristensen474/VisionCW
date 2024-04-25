from dataclasses import dataclass
from src.task3.task3 import TemplateImageKeypointMatch, apply_homography_transform, Point, Homography

import numpy as np
import random

@dataclass
class PointMatch:
    template_point: Point
    image_point: Point

class Ransac:

    def __init__(self, distance_threshold: float = 10):
        self.distance_threshold = distance_threshold
        self.sample_points_num = 4

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
        H = Vt[-1, :].reshape(3, 3).astype(np.float32)

        # Normalize the solution to ensure H[2, 2] = 1
        H = H / H[2, 2]
        H = Homography(matrix=H)
        return H

    # takes the points not fitted to line for calculation
    def calculate_homography_outliers(
        self, points: list[TemplateImageKeypointMatch]
    ) -> tuple[list[PointMatch], list[PointMatch], Homography]:
        outliers = []
        inliers = []

        def calc_homography(points: list[TemplateImageKeypointMatch]) -> Homography:
            p = np.zeros((3, self.sample_points_num))
            q = np.zeros((3, self.sample_points_num))

            for i, point in enumerate(points):  # should be 4 points in list
                p[0, i] = point.template_point.x
                p[1, i] = point.template_point.y
                p[2, i] = 1

                q[0, i] = point.image_point.x
                q[1, i] = point.image_point.y
                q[2, i] = 1

            return self.four_point_algorithm(p, q, self.sample_points_num)

        def apply_homography(points: list[TemplateImageKeypointMatch], homography: Homography) -> list[Point]:
            points_to_transform = np.array([[point.template_point.x, point.template_point.y] for point in points])
            
            return apply_homography_transform(homography, points_to_transform)


        sampled_points, unsampled_points = self.sample_points(points, self.sample_points_num)
        homography = calc_homography(sampled_points)
        if len(unsampled_points) > 0:
            transformed_points = apply_homography(unsampled_points, homography)
        else:
            transformed_points = []    

        def calculate_distance_between_points(point1: Point, point2: Point) -> np.float32:
            return np.sqrt((point1.x - point2.x) ** 2) + ((point1.y - point2.y) ** 2)
        
        for index, point in enumerate(transformed_points):
            point2 = unsampled_points[index].image_point
            distance = calculate_distance_between_points(point, point2)

            if distance > self.distance_threshold:
                outliers.append(PointMatch(template_point=unsampled_points[index].template_point,
                                           image_point=unsampled_points[index].image_point))
            else:
                inliers.append(PointMatch(template_point=unsampled_points[index].template_point,  
                                          image_point=unsampled_points[index].image_point))
                
        for point in sampled_points:
            inliers.append(PointMatch(template_point=point.template_point, image_point=point.image_point))

        return inliers, outliers, homography

    # random sampling of points for line fitting and inlier outlier calculation
    def sample_points(self, points: list[TemplateImageKeypointMatch], number: int) -> tuple[list[TemplateImageKeypointMatch], list[TemplateImageKeypointMatch]]:
        sampled = random.sample(points, number)
        unsampled = [point for point in points if point not in sampled]

        return sampled, unsampled # random choice of 4 points
    
    def refine_homography(self, inliers: list[PointMatch]) -> Homography:
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
    
    def recalc_homography_from_sampled_inliers(self, inliers: list[PointMatch]) -> tuple[Homography, list[PointMatch]]:
        p = np.zeros((3, self.sample_points_num))
        q = np.zeros((3, self.sample_points_num))

        sampled_inliers = random.sample(inliers, self.sample_points_num)

        for i, point in enumerate(sampled_inliers):
            p[0, i] = point.template_point.x
            p[1, i] = point.template_point.y
            p[2, i] = 1

            q[0, i] = point.image_point.x
            q[1, i] = point.image_point.y
            q[2, i] = 1

        return self.four_point_algorithm(p, q), sampled_inliers



    def run_ransac(self, points: list[TemplateImageKeypointMatch], iterations=100, maxRatio=0.5) -> tuple[Homography, list[PointMatch], list[PointMatch], list[PointMatch]]:

        best_inlier_count = 0
        best_outliers = []
        best_inliers = []
        best_homography = None
        for i in range(iterations):
            inliers, outliers, homography = self.calculate_homography_outliers(points)
            inlier_count = len(inliers)

            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_inliers = inliers
                best_outliers = outliers
                best_homography = homography
        
        if not best_homography:  # if empty arry or best line is none
            raise ValueError("no homography found")
        
        return self.refine_homography(best_inliers), best_inliers, best_outliers, []

        # best_homography, sampled_inliers = self.recalc_homography_from_sampled_inliers(best_inliers)
        # return best_homography, best_inliers, best_outliers, sampled_inliers
    
        # return best_homography, best_inliers, best_outliers, []
