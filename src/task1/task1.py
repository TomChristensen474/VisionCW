import math
from pathlib import Path
from dataclasses import dataclass
from typing import Iterator

import cv2 as cv
import numpy as np
import pandas as pd

Image = np.ndarray


def render(image: Image):
    cv.imshow("image", image)
    cv.waitKey(0)


def task1(folderName: str) -> float:
    this_file = Path(__file__)
    datasets_folder = this_file.parent.parent.parent / "datasets"
    dataset_folder = datasets_folder / folderName

    list_file = dataset_folder / "list.txt"
    image_files = pd.read_csv(list_file)

    total_error = 0

    for filename, actual_angle in image_files.values:
        image: Path = Path(folderName) / filename
        # if image.stem != "image5":
        # continue

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

    # 0.5 run canny on the image as a preprocessing step
    image = cannyify_image(image)
    # render(image)

    # 1. get hough lines + segments from image
    segments = hough_segments(image)

    # 2. get angle from segments
    angle = get_angle_from_segments(segments[0], segments[1])

    return angle


def cannyify_image(image: Image) -> Image:
    return cv.Canny(image, 100, 200)


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
    votes: "HoughVotes"


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

    def line(self) -> HoughLine:
        return HoughLine(self.rho, self.theta)


@dataclass
class HoughVotes:
    votes: list[HoughVote]

    def add(self, vote: HoughVote):
        self.votes.append(vote)

    @property
    def count(self) -> int:
        return len(self.votes)

    def avg_rho(self) -> float:
        return sum(vote.rho for vote in self.votes) / self.count

    def avg_theta(self) -> float:
        return sum(vote.theta for vote in self.votes) / self.count

    def point_furthest_from(self, point: "Point") -> HoughVote:
        return max(self.votes, key=lambda vote: (point.x - vote.point.x) ** 2 + (point.y - vote.point.y) ** 2)

    def segment(self, intersection: "Point") -> Segment:
        tip = self.point_furthest_from(intersection)
        base = self.point_furthest_from(tip.point)

        segment = Segment(base.point.x, base.point.y, tip.point.x, tip.point.y, self)
        return segment

    def avg_segment(self, intersection: "Point") -> Segment:
        tip = self.point_furthest_from(intersection)

        sorted_votes = sorted(self.votes, key=lambda vote: vote.point.squared_distance_from(tip.point))

        left = sorted_votes[: len(sorted_votes) // 2]
        right = sorted_votes[len(sorted_votes) // 2 :]

        segments = []
        for left_vote, right_vote in zip(left, right):
            segments.append(Segment(left_vote.point.x, left_vote.point.y, right_vote.point.x, right_vote.point.y, self))

        left_avg_x = sum(seg.x1 for seg in segments) / len(segments)
        left_avg_y = sum(seg.y1 for seg in segments) / len(segments)
        right_avg_x = sum(seg.x2 for seg in segments) / len(segments)
        right_avg_y = sum(seg.y2 for seg in segments) / len(segments)

        return Segment(left_avg_x, left_avg_y, right_avg_x, right_avg_y, self)

        # unit_segment = Segment(0, 0, 0, 1, self)
        # thetas = [get_angle_from_segments(seg, unit_segment) for seg in segments]
        # avg_theta


@dataclass
class HoughLocalMaximum:
    avg_rho: float
    avg_theta: float
    rho_idx: int
    theta_idx: int
    votes: HoughVotes

    def __init__(self, rho_idx: int, theta_idx: int, votes: HoughVotes):
        self.rho_idx = rho_idx
        self.theta_idx = theta_idx
        self.votes = votes

        self.avg_rho = votes.avg_rho()
        self.avg_theta = votes.avg_theta()

    def is_parallel_to(self, other: "HoughLocalMaximum") -> bool:
        return abs(self.avg_theta - other.avg_theta) < 5

    def hough_line(self) -> HoughLine:
        return HoughLine(self.avg_rho, self.avg_theta)


class HoughAccumulator:
    def __init__(self, rhos: np.ndarray, thetas: np.ndarray):
        self.rhos = rhos
        self.thetas = thetas

        self.matrix: list[list[HoughVotes]] = [
            [HoughVotes([]) for _ in range(len(thetas))] for _ in range(len(rhos))
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

        self.matrix[rho_idx][theta_idx].add(vote)

    def accumulator_array(self) -> np.ndarray:
        arr = np.array([[points.count for points in row] for row in self.matrix])

        # normalize to 0-255
        max_brightness = float(arr.max())
        arr = arr / (max_brightness)
        arr = arr * 255
        arr = arr.astype(np.uint8)

        return arr

    def local_maxima(self, n: int = 2) -> list["HoughLocalMaximum"]:
        # divide the accumulator into a grid of cells
        CELLS_PER_DIM = 10
        CELL_RHO_SIZE = len(self.rhos) // CELLS_PER_DIM
        CELL_THETA_SIZE = len(self.thetas) // CELLS_PER_DIM

        accumulator = self.accumulator_array()
        local_maxima: list[HoughLocalMaximum] = []

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
                votes: HoughVotes = self.matrix[rho_idx][theta_idx]

                local_maximum = HoughLocalMaximum(rho_idx, theta_idx, votes)
                local_maxima.append(local_maximum)

        local_maxima: list[HoughLocalMaximum] = sorted(
            local_maxima, key=lambda x: x.votes.count, reverse=True
        )

        # handle the case where the cell boundary crosses a hough bright point,
        # so both local maxima refer to the same hough line
        # i.e. they are right next to each other, next to the cell boundary
        MIN_DISTANCE = min(CELL_RHO_SIZE, CELL_THETA_SIZE)

        n_local_maxima: list[HoughLocalMaximum] = []
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
                    if local_maximum.votes.count > other_maximum.votes.count:
                        n_local_maxima[i] = local_maximum  # we win

                    add_to_maxima = False
                    break

            if add_to_maxima:
                n_local_maxima.append(local_maximum)

                if len(n_local_maxima) >= n:
                    break

        # we got the N brightest spots. now look around them
        # and average the rhos/thetas of the points in the neighbourhood
        NEIGHBOURHOOD_SIZE = 10
        THRESHOLD_RATIO = 0.5  # only look at lines 0.7x as bright as the brightest line

        avg_local_maxima: list[HoughLocalMaximum] = []
        for local_maximum in n_local_maxima:
            votes = HoughVotes([])
            # rho_idx = local_maximum.rho_idx
            # theta_idx = local_maximum.theta_idx

            for rho_idx in range(
                max(0, local_maximum.rho_idx - NEIGHBOURHOOD_SIZE),
                min(local_maximum.rho_idx + NEIGHBOURHOOD_SIZE, len(self.rhos) - 1),
            ):

                for theta_idx in range(
                    max(0, local_maximum.theta_idx - NEIGHBOURHOOD_SIZE),
                    min(local_maximum.theta_idx + NEIGHBOURHOOD_SIZE, len(self.thetas) - 1),
                ):

                    if self.matrix[rho_idx][theta_idx].count / local_maximum.votes.count < THRESHOLD_RATIO:
                        continue

                    for vote in self.matrix[rho_idx][theta_idx].votes:
                        votes.add(vote)

            local_maximum = HoughLocalMaximum(local_maximum.rho_idx, local_maximum.theta_idx, votes)
            avg_local_maxima.append(local_maximum)

        # return top N bright spots
        return avg_local_maxima


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

    most_voted_points = accumulator.local_maxima(2)  # i.e. both segments
    assert len(most_voted_points) == 2

    # tmp
    theta1 = refine_theta(image, most_voted_points[0].votes)
    theta2 = refine_theta(image, most_voted_points[1].votes)
    print(theta1, theta2, abs(theta1 - theta2))

    # render accumulator and most voted points
    # accumulator_img = accumulator.accumulator_array()
    # for point in most_voted_points:
    #     rho_idx = accumulator.rho_idx(point.avg_rho)
    #     theta_idx = accumulator.theta_idx(point.avg_theta)
    #     cv.circle(accumulator_img, (int(theta_idx), int(rho_idx)), 5, (255, 255, 255), 5)
    # cv.imshow("accumulator", accumulator_img)
    # cv.waitKey(0)

    # get intersection point
    line1 = most_voted_points[0].hough_line()
    line2 = most_voted_points[1].hough_line()
    intersection = get_intersection_point(line1, line2)

    # points_in_common = set([vote.point for vote in most_voted_points[0].votes]) & set(
    #     [vote.point for vote in most_voted_points[1].votes]
    # )
    # intersection_x = np.mean([point.x for point in points_in_common])
    # intersection_y = np.mean([point.y for point in points_in_common])

    # get other end of both segments
    # segment_tips: list[HoughVote] = []

    # for each segment
    # for most_voted_point in most_voted_points:

    # # sort the points on this segment by distance
    # sorted_points_on_segment = sorted(
    #     most_voted_point.votes, key=distance_from_intersection, reverse=True
    # )

    # # get the average of the 5 furthest points
    # tip_x = np.mean([vote.point.x for vote in sorted_points_on_segment[:20]])
    # tip_y = np.mean([vote.point.y for vote in sorted_points_on_segment[:20]])

    # segment_tip = most_voted_point.votes.point_furthest_from(intersection)
    # segment_tips.append(segment_tip)

    # points_further_from_tip = [
    #     segment_tip
    #     for segment_tip in segment_tips
    # ]

    # segments: list[Segment] = []
    # for segment_tip_x, segment_tip_y in segment_tips:
    #     segment = Segment(
    #         float(intersection_x),
    #         float(intersection_y),
    #         float(segment_tip_x),
    #         float(segment_tip_y),
    #     )
    #     segments.append(segment)

    segments = [most_voted_point.votes.segment(intersection) for most_voted_point in most_voted_points]

    # tmp: draw accumulator and segment tip points
    dbg_img = image.copy()
    for segment in segments:
        cv.circle(dbg_img, (int(segment.x1), int(segment.y1)), 2, (255, 255, 255), 1)
        cv.circle(dbg_img, (int(segment.x2), int(segment.y2)), 2, (255, 255, 255), 1)

    cv.imshow("accumulator", dbg_img)
    cv.waitKey(0)

    return segments


def get_intersection_point(line1: HoughLine, line2: HoughLine) -> "Point":
    rho1, theta1 = line1.rho, np.deg2rad(line1.theta)
    rho2, theta2 = line2.rho, np.deg2rad(line2.theta)

    # Check for parallel lines
    if np.abs(theta1 - theta2) < 1e-2:
        raise ValueError("Lines are parallel")

    a1 = np.cos(theta1)
    b1 = np.sin(theta1)
    c1 = rho1
    a2 = np.cos(theta2)
    b2 = np.sin(theta2)
    c2 = rho2

    # Solve for intersection point
    denominator = a1 * b2 - a2 * b1
    if np.abs(denominator) < 1e-6:
        raise ValueError("Lines are parallel 2")

    x = (c1 * b2 - c2 * b1) / denominator
    y = (a1 * c2 - a2 * c1) / denominator
    return Point(x, y)


@dataclass
class Point:
    x: int
    y: int

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    def squared_distance_from(self, other: "Point") -> float:
        return (self.x - other.x) ** 2 + (self.y - other.y) ** 2

    def distance_from(self, other: "Point") -> float:
        return math.sqrt(self.squared_distance_from(other))


def white_points(image: Image) -> Iterator[Point]:
    """Find all white pixels in the image."""
    is_white = image > 230
    ys, xs = np.nonzero(is_white)
    for x, y in zip(xs, ys):
        yield Point(x, y)


# def get_segment_from_line(image: Image, line: HoughLine) -> Segment:
#     points = list(white_points(image))

#     white_points_on_line = []
#     for point in points:
#         if line.point_is_on_line(point):
#             white_points_on_line.append(point)

#     assert len(white_points_on_line) > 0

#     # sort first on y, then on x. sort is stable so points with same x will keep
#     # their y sorted order.
#     white_points_on_line = sorted(white_points_on_line, key=lambda l: l.y)
#     white_points_on_line = sorted(white_points_on_line, key=lambda l: l.x)

#     x0 = white_points_on_line[0].x
#     y0 = white_points_on_line[0].y
#     x1 = white_points_on_line[-1].x
#     y1 = white_points_on_line[-1].y

#     return Segment(x0, y0, x1, y1)


def get_angle_from_vectors(seg1: Segment, seg2: Segment) -> float:
    vec1 = np.array([seg1.x2 - seg1.x1, seg1.y2 - seg1.y1])
    vec2 = np.array([seg2.x2 - seg2.x1, seg2.y2 - seg2.y1])

    radians = np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
    return np.degrees(radians)


def get_angle_from_segments(seg1: Segment, seg2: Segment) -> float:
    theta1 = seg1.votes.avg_theta()
    theta2 = seg2.votes.avg_theta()

    angle = np.abs(theta1 - theta2)

    # turn into an acute anvle
    angle = angle if angle < 90 else 180 - angle

    is_obtuse = get_angle_from_vectors(seg1, seg2) > 90
    return angle if not is_obtuse else 180 - angle


def get_angle_from_votes(votes1: HoughVotes, votes2: HoughVotes) -> float:
    intersection = Point(0, 0)
    segment1 = votes1.avg_segment(intersection)
    segment2 = votes2.avg_segment(intersection)

    return get_angle_from_vectors(segment1, segment2)

def refine_theta(image: Image, votes: HoughVotes, n=1000) -> float:
    thetas = [vote.theta for vote in votes.votes]
    max_theta = max(thetas)
    min_theta = min(thetas)

    diagonal_length = int(np.sqrt(image.shape[0] ** 2 + image.shape[1] ** 2))

    thetas = np.linspace(min_theta, max_theta, n)
    rhos = np.linspace(-diagonal_length, diagonal_length, 1000)

    sin_thetas = np.sin(np.deg2rad(thetas))
    cos_thetas = np.cos(np.deg2rad(thetas))

    # accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.uint8)
    accumulator = HoughAccumulator(rhos, thetas)

    for vote in votes.votes:
        point = vote.point
        x = point.x
        y = point.y

        for theta_idx, theta in enumerate(thetas):

            rho = x * cos_thetas[theta_idx] + y * sin_thetas[theta_idx]
            # rho_idx = np.abs(rhos - rho).argmin()

            # accumulator[rho_idx, theta_idx] += 1
            accumulator.add_vote(point, rho, theta)

    most_voted_point = accumulator.local_maxima(1)[0]

    return most_voted_point.avg_theta

if __name__ == "__main__":
    task1("Task1Dataset")
