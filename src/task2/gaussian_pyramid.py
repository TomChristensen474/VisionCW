from typing import Iterator

import cv2 as cv
import numpy as np


Image = np.ndarray


class PyramidLevel:
    def __init__(self, image: Image):
        self.image = image
        self.scales: dict[float, Image] = {}

    def scaled(self, scale: float) -> Image:
        if scale == 1:
            return self.image
        
        if scale in self.scales:
            return self.scales[scale]

        scaled = cv.resize(self.image, None, fx=scale, fy=scale, interpolation=cv.INTER_LINEAR)
        self.scales[scale] = scaled

        return scaled


class GaussianPyramid:
    def __init__(self, image: Image, n: int):
        self.image = image

        first_level = PyramidLevel(image)
        self.pyramid: list[PyramidLevel] = [first_level]

        for _ in range(1, n):
            next_image = cv.pyrDown(self.pyramid[-1].image)
            next_level = PyramidLevel(next_image)
            self.pyramid.append(next_level)

    def __getitem__(self, index) -> PyramidLevel:
        return self.pyramid[index]
    
    def __iter__(self) -> Iterator[PyramidLevel]:
        return iter(self.pyramid)
    
    def __len__(self) -> int:
        return len(self.pyramid)
        
