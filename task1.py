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
        # if image.stem != "image3":
            # continue

        predicted_angle = get_angle(image)
        error = abs(predicted_angle - actual_angle)
        
        print(f"Image: {image}, Predicted angle: {predicted_angle:.2f}, Actual: {actual_angle}, Error: {error:.2f}")

        total_error += error

    return total_error

def get_angle(image_path: Path) -> float:
    # 0. read image
    image = read_image(image_path)
    
    # 1. get hough lines from image
    lines = hough_lines(image)

    # 2. get segments from hough lines
    segments = [get_segment_from_line(image, line) for line in lines]

    # 3. get angle from segments
    angle = get_angle_from_segments(segments[0], segments[1])

    return angle

Image = np.ndarray

def read_image(path: Path) -> Image:
    img = cv.imread(str(path))

    # convert to greyscale i.e. collapse into one channel
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    # img is now a 2d array, with values either 51 (grey) or 255 (white)
    return img

@dataclass
class Segment:
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass
class HoughLine:
    rho: float
    theta: float # degrees

    def point_at(self, x: float) -> float:
        theta = np.deg2rad(self.theta)
        y = (self.rho - x * np.cos(theta)) / (np.sin(theta) + 0.0001)

        return y
    
    def point_is_on_line(self, point: "Point") -> bool:
        theta = np.deg2rad(self.theta)
        predicted_rho = point.x * np.cos(theta) + point.y * np.sin(theta)

        return abs(predicted_rho - self.rho) < 3

def hough_lines(image: Image) -> list[HoughLine]:
    # get all white pixels in the image
    points = list(white_points(image))

    diagonal_length = int(np.sqrt(image.shape[0] ** 2 + image.shape[1] ** 2))

    rhos = np.linspace(-diagonal_length, diagonal_length, 1000)
    thetas = np.linspace(-90, 90, 360, endpoint=False)

    sin_thetas = np.sin(np.deg2rad(thetas))
    cos_thetas = np.cos(np.deg2rad(thetas))

    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.uint8)

    for point in points:
        x = point.x
        y = point.y
        
        for theta_idx, theta in enumerate(thetas):

            rho = x * cos_thetas[theta_idx] + y * sin_thetas[theta_idx]
            rho_idx = np.abs(rhos - rho).argmin()

            accumulator[rho_idx, theta_idx] += 1

    bright_spots = find_local_maxima(accumulator, 2)
    # print(bright_spots)

    lines = []
    for bright_spot in bright_spots:
        rho_idx = bright_spot.y
        theta_idx = bright_spot.x

        rho = rhos[rho_idx]
        theta = thetas[theta_idx]

        line = HoughLine(rho, theta)
        lines.append(line)

    for line in lines:
        rho = line.rho
        theta = np.deg2rad(line.theta)

        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    cv.imshow("image", image)
    cv.waitKey(0)

    return lines


@dataclass
class Point:
    x: int
    y: int

def white_points(image: Image) -> Iterator[Point]:
    """Find all white pixels in the image."""
    is_white = image > 100
    ys, xs = np.nonzero(is_white)
    for x, y in zip(xs, ys):
        yield Point(x, y)

@dataclass
class BrightSpot:
    x: int
    y: int
    brightness: np.ndarray
    
def find_local_maxima(accumulator: np.ndarray, n: int =2) -> list[BrightSpot]:
    # divide the accumulator into a grid of cells
    CELL_SIZE = 50

    bright_spots = []

    for y in range(0, accumulator.shape[0], CELL_SIZE):
        for x in range(0, accumulator.shape[1], CELL_SIZE):
            # get the cell
            cell = accumulator[y:y+CELL_SIZE, x:x+CELL_SIZE]

            # find the maxima in the cell
            maxima = np.argwhere(cell == cell.max())[0]

            bright_spot = BrightSpot(x=x + maxima[1], y=y + maxima[0], brightness=cell[maxima[0], maxima[1]])
            bright_spots.append(bright_spot)

    # sort the cells data
    bright_spots = sorted(bright_spots, key=lambda x: x.brightness, reverse=True)

    # for bright_spot in bright_spots[:2]:
        # cv.circle(accumulator, (bright_spot.x, bright_spot.y), 5, 255, -1)
    # cv.imshow("image", accumulator)
    # cv.waitKey(0)

    # return top N bright spots
    return bright_spots[:n]


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

    radians = np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
    return np.degrees(radians)

if __name__ == "__main__":
    task1("Task1Dataset")