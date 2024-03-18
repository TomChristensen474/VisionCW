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
        image = Path(folderName) / filename
        predicted_angle = get_angle(image)
        print(f"Image: {image}, Predicted angle: {predicted_angle}, Actual angle: {actual_angle}")

        error = abs(predicted_angle - actual_angle)
        total_error += error

        return

    return total_error

def get_angle(image: Path) -> float:
    # 0. read image
    img = read_image(image)
    get_lines(img)

    return 0

Image = np.ndarray

def read_image(path: Path) -> Image:
    img = cv.imread(str(path))

    # convert to greyscale i.e. keep only one channel
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    # img is now a 2d array, with values either 51 (grey) or 255 (white)
    return img

@dataclass
class Segment:
    x1: int
    y1: int
    x2: int
    y2: int


def get_lines(image: Image) -> list[Segment]:
    hough_lines1 = hough_lines(image)
    print(hough_lines1)

    render(hough_lines1, image)

    return hough_lines1

def render(lines: list[Segment], image: Image):
    

    for line in lines:
        rho = line.rho
        theta = line.theta

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

@dataclass
class HoughLine:
    rho: float
    theta: float

def hough_lines(image: Image) -> list[HoughLine]:
    # get all white pixels in the image
    points = white_points(image)

    import math
    thetas = np.deg2rad(np.arange(-90.0, 90.0, 0.1))
    width, height = image.shape
    diag_len = int(round(math.sqrt(width * width + height * height)))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2)

    # Cache some resuable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    # Hough accumulator array of theta vs rho
    accumulator = np.zeros((2 * diag_len, num_thetas), dtype=np.uint8)
    # (row, col) indexes to edges
    are_edges = image > 60
    y_idxs, x_idxs = np.nonzero(are_edges)

    # Vote in the hough accumulator
    # for i in range(len(x_idxs)):
    for point in points:
        # x = x_idxs[i]
        # y = y_idxs[i]
        x = point.x
        y = point.y

        for t_idx in range(num_thetas):
            # Calculate rho. diag_len is added for a positive index
            rho = diag_len + int(round(x * cos_t[t_idx] + y * sin_t[t_idx]))
            accumulator[rho, t_idx] += 1

    # resize accumulator using opencv
    accumulator = cv.resize(accumulator, (1000, 1000))

    cv.imshow("image", accumulator)
    cv.waitKey(0)
            
    # votes = []

    # for i in range(accumulator.shape[0]):
    #     for j in range(accumulator.shape[1]):
    #         if accumulator[i, j] > 10:
    #             rho = rhos[i]
    #             theta = thetas[j]
    #             votes.append({
    #                 "rho": rho,
    #                 "theta": theta,
    #                 "votes": accumulator[i, j]
    #              })

    # # sort votes by votes
    # votes.sort(key=lambda x: x["votes"], reverse=True)

    # # get 2 most voted lines
    # lines = [HoughLine(v["rho"], v["theta"]) for v in votes[:2]]
                
    kmeans_result = kmeans(accumulator, 2)

    return lines


@dataclass
class Point:
    x: int
    y: int

def white_points(image: Image) -> Iterator[Point]:
    """Find all white pixels in the image."""
    xs, ys = np.where(image == 255)
    for x, y in zip(xs, ys):
        yield Point(x, y)


def kmeans(accumulator: np.ndarray, number_of_lines: int) -> list[Point]:
    # convert to float32
    accumulator = accumulator.astype(np.float32)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, _, centers = cv.kmeans(accumulator, number_of_lines, accumulator, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    breakpoint()


if __name__ == "__main__":
    task1("Task1Dataset")