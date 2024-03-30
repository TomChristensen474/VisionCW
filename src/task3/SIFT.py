from dataclasses import dataclass
from skimage.feature import hog

import cv2 as cv
import numpy as np

Image = np.ndarray


@dataclass
class Gradient:
    direction: float
    magnitude: float


@dataclass
class Descriptor:
    directions: list


@dataclass
class Point:
    x: int
    y: int


@dataclass
class Feature:
    centre: Point
    descriptor: Descriptor


@dataclass
class ScaleSpace:
    octaves: list[list[Image]]


class SIFT:
    features: list[Feature] | None
    scale_space: ScaleSpace
    kernel_size: int = 3

    def __init__(self, image: Image):
        self.image = image

        self.scale_space_construct()

        self.detect_keypoints()

    def scale_space_construct(self) -> None:
        k = 2 ** (1 / 3)
        sigma = 1.6
        scales = [sigma, k * sigma, 2 * sigma, 2 * k * sigma, 2 * (k**2) * sigma]

        image = self.image
        self.scale_space = ScaleSpace(octaves=[])
        for octave in range(4):
            octave_images = []
            for i, scale in enumerate(scales[1:]):
                sigma1 = scales[i - 1]
                sigma2 = scales[i]
                difference_of_gaussian = self.difference_of_gaussian(
                    image, sigma1, sigma2
                )
                octave_images.append(difference_of_gaussian)

            self.scale_space.octaves.append(octave_images)
            image = cv.resize(image, (image.shape[1] // 2, image.shape[0] // 2))

    def difference_of_gaussian(
        self, image: Image, sigma1: float, sigma2: float
    ) -> Image:
        img1 = cv.GaussianBlur(
            image, (self.kernel_size, self.kernel_size), sigmaX=sigma1
        )
        img2 = cv.GaussianBlur(
            image, (self.kernel_size, self.kernel_size), sigmaX=sigma2
        )

        difference_of_gaussian = cv.subtract(img1, img2)
        return difference_of_gaussian

    def detect_keypoints(self):
        for octave in self.scale_space.octaves:
            for i, _ in enumerate(octave[1:-1]):
                self.detect_extrema(octave[i - 1 : i + 1])
        pass

    def detect_extrema(self, scales: list[Image]) -> list[Feature]:
        features = []
        for i in range(1, scales[1].shape[0] - 1):
            for j in range(1, scales[1].shape[1] - 1):
                if self.is_extreme(scales, i, j):
                    descriptor = self.describe_keypoints(scales[1], i, j)
                    features.append(
                        Feature(
                            centre=Point(i, j), descriptor=Descriptor(directions=[])
                        )
                    )

        return features

    def is_extreme(self, scales: list[Image], i: int, j: int) -> bool:
        threshold = 0.03  # TODO find a good threshold
        maximum = np.max(
            scales[0][i - 1 : i + 1, j - 1 : j + 1],
            scales[1][i - 1 : i + 1, j - 1 : j + 1],
            scales[2][i - 1 : i + 1, j - 1 : j + 1],
        )
        minimum = np.min(
            scales[0][i - 1 : i + 1, j - 1 : j + 1],
            scales[1][i - 1 : i + 1, j - 1 : j + 1],
            scales[2][i - 1 : i + 1, j - 1 : j + 1],
        )

        if (
            scales[1][i, j] > maximum + threshold
            or scales[1][i, j] < minimum - threshold
        ):
            return False

        return False

    def describe_keypoints(self, scale, i, j) -> Descriptor:
        directions = hog(
            scale,
            orientations=8,
            pixels_per_cell=(16, 16),
            cells_per_block=(4, 4),
            visualize=False,
            multichannel=True,
        )

        # directions = []
        # for x in range(i - 8, i + 8, 4):
        #     for y in range(j - 8, j + 8, 4):
        #         patch = scale[x : x + 4, y : y + 4]

        # # calculate the gradient direction and magnitude for each pixel in the patch
        # gradients = []
        # for row in patch:
        #     gradient_row = []
        #     for pixel in row:
        #         dx = cv.Sobel(pixel, cv.CV_64F, 1, 0, ksize=3)
        #         dy = cv.Sobel(pixel, cv.CV_64F, 0, 1, ksize=3)
        #         magnitude = np.sqrt(dx**2 + dy**2)
        #         direction = np.arctan2(dy, dx)
        #         gradient_row.append(Gradient(direction=direction, magnitude=magnitude))
        #     gradients.append(gradient_row)

        # calculate the histogram

        if len(directions) > 128 or len(directions) < 128:
            raise ValueError("list should be 128 length")

        return Descriptor(directions=directions)

    def match_features(self, other_features: list[Feature]):
        pass
