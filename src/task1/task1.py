import math
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Iterator

import cv2 as cv
import numpy as np
import pandas as pd
from joblib import delayed, Parallel

from thinner import Thinner

Image = np.ndarray


@dataclass
class Task1Config:
    debug_level: int = 0
    multithreaded: bool = True
    cannyify_image: bool = False
    thin_image: bool = True
    refined_votes: bool = False
    n_average_segment_tips: int = 1
    trim_segment_edges: int = 0
    manysegs_average_segments: bool = True
    use_average_theta: bool = False
    faster_nearest_idx: bool = True

    def is_debug(self, level: int) -> bool:
        return self.debug_level >= level


config = Task1Config()


class CsvWriter:
    def __init__(self):
        self.path = Path(__file__).parent / "results.csv"

        self.config_columns = list(Task1Config.__dataclass_fields__.keys())
        self.config_columns.remove("debug_level")

        self.columns = [
            "when",
            "total_error",
            "average_error",
            "avg_runtime",
        ] + self.config_columns

        if not self.path.exists():  # if csv file doesn't exist, create it and write row names (first line)
            self.file = open(self.path, "w")
            self.file.write(",".join(self.columns) + "\n")
        else:  # if csv file exists, make sure row names are what we are going to write
            with self.path.open("r") as f:
                expected_header = ",".join(self.columns).replace(" ", "")
                actual_header = f.readline().strip().replace(" ", "")
                assert (
                    expected_header == actual_header
                ), f"CSV file has wrong header\nexpecting: {expected_header}\ngot:       {actual_header}"

            self.file = open(self.path, "a")

    def add(self, total_error, average_error, avg_runtime):
        when = time.strftime("%Y-%m-%d %H:%M:%S")
        fields_to_write = [when, total_error, average_error, avg_runtime]
        fields_to_write += [getattr(config, field) for field in self.config_columns]

        fields_to_write = [f"{x:.2f}" if isinstance(x, float) else str(x) for x in fields_to_write]
        self.file.write(",".join(fields_to_write) + "\n")


def render(image: Image, wait=True):
    cv.imshow("image", image)
    cv.waitKey(0 if wait else 1)


def task1(folderName: str) -> float:
    this_file = Path(__file__)
    datasets_folder = this_file.parent.parent.parent / "datasets"
    dataset_folder = datasets_folder / folderName

    list_file = dataset_folder / "list.txt"
    image_files = pd.read_csv(list_file)

    # for filename, actual_angle in image_files.values:
    @delayed
    def measure_angle(filename):
        image: Path = dataset_folder / filename

        start = time.time()
        predicted_angle = get_angle(image)
        end = time.time()

        runtime = end - start
        return predicted_angle, runtime

    n_jobs = -1 if config.multithreaded else 1
    measured_angles_generator = Parallel(n_jobs=n_jobs, return_as="generator")(
        measure_angle(filename) for filename, _ in image_files.values
    )

    total_error = 0
    total_runtime = 0
    for (filename, actual_angle), (predicted_angle, runtime) in zip(  # type: ignore
        image_files.values, measured_angles_generator
    ):
        image: Path = dataset_folder / filename

        error = abs(predicted_angle - actual_angle)

        total_error += error
        total_runtime += runtime

        print(
            f"{image.stem} | Predicted angle: {predicted_angle:.2f}, Actual: {actual_angle}, Error: {error:.2f} | Runtime: {runtime:.2f}s"
        )

    average_error = total_error / len(image_files)
    average_runtime = total_runtime / len(image_files)
    print(
        f"Total error: {total_error:.2f}, Average error: {average_error:.2f}, Average runtime: {average_runtime:.2f}s"
    )

    csv_writer = CsvWriter()
    csv_writer.add(total_error, average_error, average_runtime)

    return total_error


def get_angle(image_path: Path) -> float:
    # 0. read image
    image = read_image(image_path)

    if config.cannyify_image:
        # run canny on the image as a preprocessing step
        image = cannyify_image(image)
    elif config.thin_image:
        _, binary_image = cv.threshold(image, 120, 255, cv.THRESH_BINARY)
        thinner = Thinner()
        image = thinner.thin_image(binary_image)

    if config.is_debug(1):
        render(image)

    # 1. get hough lines + segments from image
    segments = hough_segments(image)

    # 2. get angle from segments
    if isinstance(segments, float):
        angle = segments
    else:
        assert isinstance(segments, list)

        if config.use_average_theta:
            angle = get_angle_from_segments(segments[0], segments[1])
        else:
            angle = get_angle_from_vectors(segments[0], segments[1])

    return angle


def cannyify_image(image: Image) -> Image:
    return cv.Canny(image, 100, 200)


def read_image(path: Path) -> Image:
    img = cv.imread(str(path))

    # convert to greyscale i.e. collapse into one channel
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    # img is now a 2d array, with values either 51 (grey) or 255 (white)
    return img


@dataclass
class Point:
    x: int | float
    y: int | float

    def squared_distance_from(self, other: "Point") -> float:
        return (self.x - other.x) ** 2 + (self.y - other.y) ** 2

    def distance_from(self, other: "Point") -> float:
        return math.sqrt(self.squared_distance_from(other))

    def __hash__(self) -> int:
        return hash((self.x, self.y))


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

    def avg_point(self) -> "Point":
        avg_x = sum(vote.point.x for vote in self.votes) / self.count
        avg_y = sum(vote.point.y for vote in self.votes) / self.count
        return Point(avg_x, avg_y)

    def point_furthest_from(self, point: "Point") -> HoughVote:
        return max(self.votes, key=lambda vote: point.squared_distance_from(vote.point))

    def n_points_furthest_from(self, point: "Point", n: int, trim: int) -> "HoughVotes":
        # the segment edges can be funny, so trim says "ignore the 3 furthest points"
        # assuming they're in the "funny" region
        n += trim

        if n == 1:
            votes = [self.point_furthest_from(point)]
        elif n > len(self.votes):
            # no need to sort to get furthest, if we already have
            # more than n points
            votes = self.votes.copy()
        else:
            votes = sorted(self.votes, key=lambda vote: point.squared_distance_from(vote.point), reverse=True)
            votes = votes[:n]

        votes = votes[trim:]
        return HoughVotes(votes)

    def avg_point_furthest_from(self, point: "Point", n: int, trim: int, median: bool = False) -> HoughVote:
        n_points = self.n_points_furthest_from(point, n, trim)
        n_votes = n_points.votes

        if median:
            return n_votes[len(n_votes) // 2]
        else:
            avg_x = sum(vote.point.x for vote in n_votes) / len(n_votes)
            avg_y = sum(vote.point.y for vote in n_votes) / len(n_votes)
            avg_point = Point(avg_x, avg_y)

            avg_rho = sum(vote.rho for vote in n_votes) / len(n_votes)
            avg_theta = sum(vote.theta for vote in n_votes) / len(n_votes)

            return HoughVote(avg_point, avg_rho, avg_theta)

    def segment(self, intersection: "Point", n: int, trim: int) -> Segment:
        avg_tip = self.avg_point_furthest_from(intersection, n, trim)
        avg_base = self.avg_point_furthest_from(avg_tip.point, n, trim)

        segment = Segment(avg_base.point.x, avg_base.point.y, avg_tip.point.x, avg_tip.point.y, self)
        return segment

    def avg_segment(self, intersection: "Point") -> Segment:
        tip = self.point_furthest_from(intersection)

        sorted_votes = sorted(self.votes, key=lambda vote: vote.point.squared_distance_from(tip.point))

        middle_idx = len(sorted_votes) // 2
        left = sorted_votes[:middle_idx]
        right = sorted_votes[middle_idx:]

        segments = []
        for left_vote, right_vote in zip(left, right):
            segments.append(
                Segment(left_vote.point.x, left_vote.point.y, right_vote.point.x, right_vote.point.y, self)
            )

        left_avg_x = sum(seg.x1 for seg in segments) / len(segments)
        left_avg_y = sum(seg.y1 for seg in segments) / len(segments)
        right_avg_x = sum(seg.x2 for seg in segments) / len(segments)
        right_avg_y = sum(seg.y2 for seg in segments) / len(segments)

        return Segment(left_avg_x, left_avg_y, right_avg_x, right_avg_y, self)


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
        if config.faster_nearest_idx:
            # https://stackoverflow.com/a/26026189
            idx = arr.searchsorted(val, side="left")
            # note: this is much faster than abs().argmin(), but
            # truncates instead of rounding. in practice this means
            # get_idx(5.9999) will return the same as get_idx(5) if
            # arr only has whole numbers.
        else:
            idx = np.abs(arr - val).argmin()

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

                # find the highest vote count of the cell
                cell_max = cell.max()
                if cell_max == 0:
                    continue  # whole cells is 0s, maxima isn't here

                # find the xy of the maximum in the cell
                max_idxs = np.argwhere(cell == cell_max)[0]
                rho_idx_in_cell = max_idxs[0]
                theta_idx_in_cell = max_idxs[1]

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
        NEIGHBOURHOOD_SIZE = 3
        THRESHOLD_RATIO = 0.2  # only look at lines 0.5x as bright as the brightest line

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


def hough_segments(image: Image) -> list[Segment] | float:
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
            accumulator.add_vote(point, rho, theta)

    most_voted_points = accumulator.local_maxima(2)  # i.e. both segments
    assert len(most_voted_points) == 2, f"Expected 2 local maxima, got {len(most_voted_points)}"

    if config.refined_votes:
        most_voted_points = [refine_votes(image, point.votes) for point in most_voted_points]
        # theta1 = refine_votes(image, most_voted_points[0].votes)
        # theta2 = refine_votes(image, most_voted_points[1].votes)
        # print(theta1, theta2, abs(theta1 - theta2))
        # return abs(theta1 - theta2)

    if config.is_debug(1):
        # render accumulator and most voted points
        accumulator_img = accumulator.accumulator_array()
        for point in most_voted_points:
            rho_idx = accumulator.rho_idx(point.avg_rho)
            theta_idx = accumulator.theta_idx(point.avg_theta)
            cv.circle(accumulator_img, (int(theta_idx), int(rho_idx)), 5, (255, 255, 255), 5)
        cv.imshow("accumulator", accumulator_img)
        cv.waitKey(0)

    # get intersection point
    line1 = most_voted_points[0].hough_line()
    line2 = most_voted_points[1].hough_line()
    intersection = get_intersection_point(line1, line2)

    n = config.n_average_segment_tips
    trim = config.trim_segment_edges
    if config.manysegs_average_segments:
        segments = [
            most_voted_point.votes.avg_segment(intersection) for most_voted_point in most_voted_points
        ]
    else:
        segments = [
            most_voted_point.votes.segment(intersection, n, trim) for most_voted_point in most_voted_points
        ]

    if config.is_debug(1):
        # draw image and segment tip points
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


def white_points(image: Image) -> Iterator[Point]:
    """Find all white pixels in the image."""
    is_white = image > 120
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

    # turn into an acute angle
    angle = angle if angle < 90 else 180 - angle

    is_obtuse = get_angle_from_vectors(seg1, seg2) > 90
    return angle if not is_obtuse else 180 - angle


def get_angle_from_votes(votes1: HoughVotes, votes2: HoughVotes) -> float:
    intersection = Point(0, 0)
    segment1 = votes1.avg_segment(intersection)
    segment2 = votes2.avg_segment(intersection)

    return get_angle_from_vectors(segment1, segment2)


def refine_votes(image: Image, votes: HoughVotes, n=100) -> HoughLocalMaximum:
    thetas = [vote.theta for vote in votes.votes]
    max_theta = max(thetas)
    min_theta = min(thetas)

    rhos = [vote.rho for vote in votes.votes]
    max_rho = max(rhos)
    min_rho = min(rhos)

    thetas = np.linspace(min_theta, max_theta, n)
    rhos = np.linspace(min_rho, max_rho, n)

    sin_thetas = np.sin(np.deg2rad(thetas))
    cos_thetas = np.cos(np.deg2rad(thetas))

    accumulator = HoughAccumulator(rhos, thetas)

    for vote in votes.votes:
        point = vote.point
        x = point.x
        y = point.y

        for theta_idx, theta in enumerate(thetas):
            rho = x * cos_thetas[theta_idx] + y * sin_thetas[theta_idx]
            accumulator.add_vote(point, rho, theta)

    most_voted_point = accumulator.local_maxima(1)[0]

    if config.is_debug(1):
        # render accumulator and most voted points
        accumulator_img = accumulator.accumulator_array()

        rho_idx = accumulator.rho_idx(most_voted_point.avg_rho)
        theta_idx = accumulator.theta_idx(most_voted_point.avg_theta)
        cv.circle(accumulator_img, (int(theta_idx), int(rho_idx)), 5, (255, 255, 255), 5)

        cv.imshow("refined accumulator", accumulator_img)
        cv.waitKey(0)

    return most_voted_point


if __name__ == "__main__":
    task1("Task1Dataset")
