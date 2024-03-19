from pathlib import Path
from dataclasses import dataclass
from typing import Iterator

import cv2 as cv
import numpy as np
import pandas as pd


def task1(folderName: str) -> float:
    list_file = Path(folderName) / "list.txt"
    image_files = pd.read_csv(list_file)

    total_error = 0

    for filename, actual_angle in image_files.values:
        image: Path = Path(folderName) / filename
        # if image.stem != "image9":
        #     continue

        predicted_angle = get_angle(image)
        error = abs(predicted_angle - actual_angle)

        print(
            f"Image: {image}, Predicted angle: {predicted_angle:.2f}, Actual: {actual_angle}, Error: {error:.2f}"
        )

        total_error += error
    print(f"Total error: {total_error:.2f}")
    return total_error


def get_angle(image_path: Path) -> float:
    # 0. read image
    image = read_image(image_path)

    # 1. get hough lines + segments from image
    segments = hough_segments(image)

    # 2. get angle from segments
    angle = get_angle_from_segments(segments[0], segments[1])

    return angle


def filter_similar_thetas(thetas):
    for theta in thetas:
        removed_thetas = []
        thetas_to_average = []
        removing = False

        for t in thetas:
            if abs(theta - t) < 5:
                removed_thetas.append(t)
                thetas_to_average.append(t)
                # thetas.remove(t)
                removing = True

            elif abs(theta - (t - 180)) < 5:
                removed_thetas.append(t)
                thetas_to_average.append(t - 180)
                removing = True

        if removing:
            mean_thetas = sum(thetas_to_average) / len(thetas_to_average)
            if mean_thetas < 0:
                mean_thetas += 180
            thetas.append(mean_thetas)
            for t in removed_thetas:
                thetas.remove(t)
    return thetas


def calculate_angle(thetas, acute):
    if acute:
        if thetas[1] - thetas[0] > 90:
            angle = 180 - (thetas[1] - thetas[0])
        else:
            angle = thetas[1] - thetas[0]
    else:
        if thetas[1] - thetas[0] > 90:
            angle = thetas[1] - thetas[0]
        else:
            angle = 180 - (thetas[1] - thetas[0])

    return abs(angle)


Image = np.ndarray


def read_image(path: Path) -> Image:
    img = cv.imread(str(path))

    # convert to greyscale i.e. collapse into one channel
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    # img is now a 2d array, with values either 51 (grey) or 255 (white)
    return img


@dataclass
class Segment:
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class HoughLine:
    rho: float
    theta: float  # degrees

    def point_at(self, x: float) -> float:
        theta = np.deg2rad(self.theta)
        y = (self.rho - x * np.cos(theta)) / (np.sin(theta) + 0.0001)

        return y

    def point_is_on_line(self, point: "Point") -> bool:
        theta = np.deg2rad(self.theta)
        predicted_rho = point.x * np.cos(theta) + point.y * np.sin(theta)

        return abs(predicted_rho - self.rho) < 3


@dataclass
class HoughVote:
    point: "Point"
    rho: float
    theta: float


@dataclass
class HoughLocalMaxima:
    rho: float
    theta: float
    rho_idx: int
    theta_idx: int
    votes: list[HoughVote]


class HoughAccumulator:
    def __init__(self, rhos: np.ndarray, thetas: np.ndarray):
        self.rhos = rhos
        self.thetas = thetas

        self.matrix: list[list[list[HoughVote]]] = [
            [[] for _ in range(len(thetas))] for _ in range(len(rhos))
        ]

    def get_idx(self, val: float, arr: np.ndarray) -> int:
        # https://stackoverflow.com/a/26026189
        idx = arr.searchsorted(val, side="left")
        # note: this is much faster than abs().argmin(), but
        # truncates instead of rounding. in practice this means
        # get_idx(5.9999) will return the same as get_idx(5) if
        # arr only has whole numbers.

        return int(idx)

    def rho_idx(self, rho: float) -> int:
        return self.get_idx(rho, self.rhos)

    def theta_idx(self, theta: float) -> int:
        return self.get_idx(theta, self.thetas)

    def add_vote(self, point: "Point", rho: float, theta: float):
        rho_idx = self.rho_idx(rho)
        theta_idx = self.theta_idx(theta)
        vote = HoughVote(point, rho, theta)

        self.matrix[rho_idx][theta_idx].append(vote)

    def accumulator_array(self) -> np.ndarray:
        arr = np.array([[len(points) for points in row] for row in self.matrix])

        # normalize to 0-255
        max_brightness = float(arr.max())
        arr = arr / (max_brightness)
        arr = arr * 255
        arr = arr.astype(np.uint8)

        return arr

    def local_maxima(self, n: int = 2) -> list["HoughLocalMaxima"]:
        # divide the accumulator into a grid of cells
        CELLS_PER_DIM = 10
        CELL_RHO_SIZE = len(self.rhos) // CELLS_PER_DIM
        CELL_THETA_SIZE = len(self.thetas) // CELLS_PER_DIM

        accumulator = self.accumulator_array()
        local_maxima: list[HoughLocalMaxima] = []

        for rho_cell_idx in range(0, accumulator.shape[0], CELL_RHO_SIZE):
            for theta_cell_idx in range(0, accumulator.shape[1], CELL_THETA_SIZE):
                # get the cell
                cell = accumulator[
                    rho_cell_idx : rho_cell_idx + CELL_RHO_SIZE,
                    theta_cell_idx : theta_cell_idx + CELL_THETA_SIZE,
                ]

                # find the xy of the maximum in the cell
                max_idxs = np.argwhere(cell == cell.max())[0]
                rho_idx_in_cell = max_idxs[0]
                theta_idx_in_cell = max_idxs[1]

                if cell[rho_idx_in_cell, theta_idx_in_cell] == 0:
                    continue  # whole cells is 0s, maxima isn't here

                rho_idx = rho_cell_idx + rho_idx_in_cell
                theta_idx = theta_cell_idx + theta_idx_in_cell

                # calculate average rho and theta of points
                votes: list[HoughVote] = self.matrix[rho_idx][theta_idx]
                rho = np.mean([vote.rho for vote in votes])
                theta = np.mean([vote.theta for vote in votes])

                local_maximum = HoughLocalMaxima(
                    float(rho), float(theta), rho_idx, theta_idx, votes
                )
                local_maxima.append(local_maximum)

        local_maxima = sorted(local_maxima, key=lambda x: len(x.votes), reverse=True)

        # handle the case where the cell boundary crosses a hough bright point,
        # so both local maxima refer to the same hough line
        # i.e. they are right next to each other, next to the cell boundary
        MIN_DISTANCE = min(CELL_RHO_SIZE, CELL_THETA_SIZE) / 8

        n_local_maxima: list[HoughLocalMaxima] = []
        for local_maximum in local_maxima:
            rho_idx = local_maximum.rho_idx
            theta_idx = local_maximum.theta_idx

            # check if this point is too close to another already in
            # n_local_maxima
            add_to_maxima = True

            for i, other_maximum in enumerate(n_local_maxima):
                other_rho_idx = other_maximum.rho_idx
                other_theta_idx = other_maximum.theta_idx

                # manhattan distance for optimisation reasons
                rho_distance = abs(rho_idx - other_rho_idx)
                theta_distance = abs(theta_idx - other_theta_idx)
                distance = rho_distance + theta_distance

                # edge case: also take into account wrap around distance
                # in theta axis, i.e. 89 and -89 are only 2 degrees apart
                theta_wraparound_distance = (
                    abs(other_theta_idx + len(self.thetas) - theta_idx)
                    if theta_idx > other_theta_idx
                    else abs(theta_idx + len(self.thetas) - other_theta_idx)
                )
                distance = min(distance, theta_wraparound_distance)

                if distance < MIN_DISTANCE:
                    # we are too close to that point, assume both we both
                    # represent the same line

                    # question is, which one of us should stay
                    if len(local_maximum.votes) > len(other_maximum.votes):
                        n_local_maxima[i] = local_maximum  # we win

                    add_to_maxima = False
                    break

            if add_to_maxima:
                n_local_maxima.append(local_maximum)

                if len(n_local_maxima) >= n:
                    break

        # return top N bright spots
        return n_local_maxima


def hough_segments(image: Image) -> list[Segment]:
    # get all white pixels in the image
    points = list(white_points(image))

    diagonal_length = int(np.sqrt(image.shape[0] ** 2 + image.shape[1] ** 2))

    rhos = np.linspace(-diagonal_length, diagonal_length, 1000)
    thetas = np.linspace(-90, 90, 1000, endpoint=False)

    # to avoid re-calculating them every loop
    sin_thetas = np.sin(np.deg2rad(thetas))
    cos_thetas = np.cos(np.deg2rad(thetas))

    # accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.uint8)
    accumulator = HoughAccumulator(rhos, thetas)

    for point in points:
        x = point.x
        y = point.y

        for theta_idx, theta in enumerate(thetas):

            rho = x * cos_thetas[theta_idx] + y * sin_thetas[theta_idx]
            # rho_idx = np.abs(rhos - rho).argmin()

            # accumulator[rho_idx, theta_idx] += 1
            accumulator.add_vote(point, rho, theta)

    # bright_spots = find_local_maxima(accumulator, 2)
    most_voted_points = accumulator.local_maxima(2)  # i.e. both segments

    # get intersection point
    points_in_common = set([vote.point for vote in most_voted_points[0].votes]) & set(
        [vote.point for vote in most_voted_points[1].votes]
    )
    intersection_x = np.mean([point.x for point in points_in_common])
    intersection_y = np.mean([point.y for point in points_in_common])

    # get other end of both segments
    points_further_from_intersection: list[tuple[float, float]] = []

    # for each segment
    for most_voted_point in most_voted_points:

        # sort the points on this segment by distance
        distance_from_intersection = lambda vote: np.sqrt(
            (vote.point.x - intersection_x) ** 2 + (vote.point.y - intersection_y) ** 2
        )
        sorted_points_on_segment = sorted(
            most_voted_point.votes, key=distance_from_intersection, reverse=True
        )

        # get the average of the 5 furthest points
        tip_x = np.mean([vote.point.x for vote in sorted_points_on_segment[:20]])
        tip_y = np.mean([vote.point.y for vote in sorted_points_on_segment[:20]])

        points_further_from_intersection.append((float(tip_x), float(tip_y)))

    segments: list[Segment] = []
    for segment_tip_x, segment_tip_y in points_further_from_intersection:
        segment = Segment(
            float(intersection_x),
            float(intersection_y),
            float(segment_tip_x),
            float(segment_tip_y),
        )
        segments.append(segment)

    return segments


@dataclass
class Point:
    x: int
    y: int

    def __hash__(self) -> int:
        return hash((self.x, self.y))


def white_points(image: Image) -> Iterator[Point]:
    """Find all white pixels in the image."""
    is_white = image > 230
    ys, xs = np.nonzero(is_white)
    for x, y in zip(xs, ys):
        yield Point(x, y)


def get_segment_from_line(image: Image, line: HoughLine) -> Segment:
    points = list(white_points(image))

    white_points_on_line = []
    for point in points:
        if line.point_is_on_line(point):
            white_points_on_line.append(point)

    assert len(white_points_on_line) > 0

    # sort first on y, then on x. sort is stable so points with same x will keep
    # their y sorted order.
    white_points_on_line = sorted(white_points_on_line, key=lambda l: l.y)
    white_points_on_line = sorted(white_points_on_line, key=lambda l: l.x)

    x0 = white_points_on_line[0].x
    y0 = white_points_on_line[0].y
    x1 = white_points_on_line[-1].x
    y1 = white_points_on_line[-1].y

    return Segment(x0, y0, x1, y1)


def get_angle_from_segments(seg1: Segment, seg2: Segment) -> float:
    vec1 = np.array([seg1.x2 - seg1.x1, seg1.y2 - seg1.y1])
    vec2 = np.array([seg2.x2 - seg2.x1, seg2.y2 - seg2.y1])

    radians = np.arccos(
        np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    )
    return np.degrees(radians)


if __name__ == "__main__":
    task1("Task1Dataset")
