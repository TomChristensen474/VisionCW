import cv2 as cv
import numpy as np


Image = np.ndarray


class PyramidLevel:
    def __init__(self, image):
        self.image = image
        self.scales = {}

    def scaled(self, scale: float):
        if scale in self.scales:
            return self.scales[scale]

        scaled = cv.resize(self.image, None, fx=scale, fy=scale, interpolation=cv.INTER_LINEAR)
        self.scales[scale] = scaled

        return scaled


class GaussianPyramid:
    def __init__(self, image: Image, n: int = 5):
        self.image = image
        self.pyramid = [image]

        for _ in range(1, n):
            next_level = cv.pyrDown(self.pyramid[-1])
            self.pyramid.append(next_level)

    def __getitem__(self, index):
        return self.pyramid[index]
    
    # def matches(self, other: "GaussianPyramid") -> float:
        
