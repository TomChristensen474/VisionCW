from dataclasses import dataclass
# from skimage.feature import hog
from tqdm import tqdm

import cv2 as cv
import numpy as np
import time

Image = np.ndarray


@dataclass
class Gradient:
    direction: float
    magnitude: float


@dataclass
class Point:
    x: int
    y: int


@dataclass
class Keypoint:
    centre: Point
    scale: Image
    orientation: float


@dataclass
class DescribedKeypoint:
    keypoint: Keypoint
    descriptor: np.ndarray


# octave 1: [[scale_1], ..., [scale_n]]
# octave 1/2: [[scale_1], ... ,[scale_n]]
# octave 1/4: [[scale_1], ..., [scale_n]]

@dataclass
class ScaleSpace:
    octaves: list[list[Image]]


class Sift:
    features: list[DescribedKeypoint]
    scale_space: ScaleSpace
    kernel_size: int = 5
    num_octaves: int = 4
    neighbourhood_size: int
    octave_scale_factor: int = 2

    def __init__(self, image: Image):
        self.image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        self.image = cv.resize(self.image, (128, 128))
        self.neighbourhood_size = 2**self.num_octaves

        self.scale_space_construct()

        self.find_keypoints()

        self.draw_keypoints()

    def draw_keypoints(self):
        print(len(self.features))
        for feature in self.features:
            cv.circle(
                self.image,
                (feature.keypoint.centre.y, feature.keypoint.centre.x),
                3,
                (0, 255, 0),
                5,
            )
        cv.imshow("keypoints", self.image)
        cv.waitKey(0)

    def scale_space_construct(self) -> None:
        k = 2**1 / 3
        sigma = 1.6
        scales = [sigma, k * sigma, 2 * sigma, 2 * k * sigma, 2 * (k**2) * sigma]
        image = self.image
        self.scale_space = ScaleSpace(octaves=[])
        for octave in range(self.num_octaves):
            octave_images = []
            for i, scale in enumerate(scales[1:]):
                sigma1 = scales[i - 1]
                sigma2 = scales[i]
                # TODO:  check if correct way round
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
        # cv.imshow("im1", img1)
        # cv.imshow("im2", img2)
        difference_of_gaussian = cv.subtract(img1, img2)
        # cv.imshow("DoG", difference_of_gaussian)
        # cv.waitKey(0)
        return difference_of_gaussian

    """
    Checks if a pixel / point is on the edge
    """
    def isPixelOnEdge(self, x, y, width, height):
        # if on first row or last row or on first column or on last column 
        if x == 0 or x == height-1 or y == 0 or y == width - 1:
            return True
        return False

    def maxValueInExtrema(self, x, y, previous_scale_space, current_scale_space, next_scale_space):
        # point x,y: is the potential keypoint we are checking for
        # [[a11 a12 a13]]
        # [[a21 (x,y) a23]]
        # [[a31 a32 a33]]
        
        maxValueInExtrema = 0
        # need to find max in previous, current and next scale space
        for i, scale_space in enumerate([previous_scale_space, current_scale_space, next_scale_space]):
            pixel_value = scale_space[x][y]

            first_row, first_col = x-1, y-1
            second_row, second_col = x,y
            third_row, third_col = x+1, y+1
            p11 = scale_space[first_row][first_col]
            p12 = scale_space[first_row][second_col]
            p13 = scale_space[first_row][third_col]
            p21 = scale_space[second_row][first_col]
            # ignore p22 (potential keypoint) for now
            p23 = scale_space[second_row][third_col]
            p31 = scale_space[third_row][first_col]
            p32 = scale_space[third_row][second_col]
            p33 = scale_space[third_row][third_col]

            # if we are at the current scale space ignore point p22 (the keypoint we are detecting)
            p22 = 0 # make key lowest possible value
            if i != 1:
                p22 = scale_space[second_row][second_col]

            pixel_values = [p11,p12,p13,p21,p22,p23,p31,p32,p33]
            maxValueInExtrema = max(maxValueInExtrema,max(pixel_values))
        
        return maxValueInExtrema


    def find_keypoint_inside_scale_space(self, previous_scale_space, current_scale_space, next_scale_space):
        features = []
        keypoints: list[Point] = []
        for x, row in enumerate(current_scale_space):
            for y, current_scale_space_pixel_value in enumerate(row):
                if not self.isPixelOnEdge(x=x,y=y,width=len(current_scale_space[0]), height=len(current_scale_space)):
                    # a22: the potential keypoint we are checking for
                    # [[a11 a12 a13]]
                    # [[a21 a22 a23]]
                    # [[a31 a32 a33]]

                    if current_scale_space_pixel_value > self.maxValueInExtrema(x,y,previous_scale_space,current_scale_space,next_scale_space):
                        keypoints.append(Point(x,y))

        
        print("keypoints")
        print(keypoints)
        exit()
                        #keypoints.append(Keypoint(centre=Point(x,y), scale=, orientation=))
        
        features.extend(self.describe_keypoints(keypoints))                    
        return features

    def find_keypoints(self):
        print("Find keypoints")
        for octave in self.scale_space.octaves:  # for each octave 
            # for each scale space in the octave apart from the first and last
            for i in range(1, len(octave)-1):  
                
                #scale_space = octave[i] 
                #print(scale_space)
                self.find_keypoint_inside_scale_space(previous_scale_space=octave[i-1],
                                                      current_scale_space=octave[i],
                                                      next_scale_space=octave[i+1])
                
                
                #exit()
                #octave_idx = i + 1
                #self.detect_extrema(octave[octave_idx - 1 : octave_idx + 2], octave_idx)  



    def detect_extrema(self, scales: list[Image], octave) -> None:
        features = []
        for i in range(1, scales[1].shape[0] - 1):
            for j in range(1, scales[1].shape[1] - 1):
                # check if pixel is a local maximum or minimum
                start = time.time()
                if self.is_extreme(scales, i, j):
                    # orientation_assigment
                    keypoints = self.get_keypoints(scales[1], i, j, octave)
                    # keypoint description
                    features.extend(self.describe_keypoints(keypoints))
                    end = time.time()
                    # print(end - start)


                
        self.features = features

    def is_extreme(self, scales: list[Image], i: int, j: int) -> bool:
        threshold = 0.0  # TODO find a good threshold
        maximum = np.max(
            [
                scales[0][i - 1 : i + 2, j - 1 : j + 2],
                scales[1][i - 1 : i + 2, j - 1 : j + 2],
                scales[2][i - 1 : i + 2, j - 1 : j + 2],
            ]
        )
        minimum = np.min(
            [
                scales[0][i - 1 : i + 2, j - 1 : j + 2],
                scales[1][i - 1 : i + 2, j - 1 : j + 2],
                scales[2][i - 1 : i + 2, j - 1 : j + 2],
            ]
        )

        if (
            scales[1][i, j] >= maximum + threshold
            or scales[1][i, j] <= minimum - threshold
        ):
            return True

        return False

    def get_keypoints(self, scale, i, j, octave) -> list[Keypoint]:
        # create histogram with 36 bins
        histogram = np.zeros(36)
        scale_factor = (1 / self.octave_scale_factor) ** octave
        neighbourhood_size = int(
            self.neighbourhood_size * (scale_factor)
        )  # 1 at smallest scale, 32 at largest
        x1 = i - neighbourhood_size
        y1 = j - neighbourhood_size
        x2 = i + neighbourhood_size
        y2 = j + neighbourhood_size
        if x1 < 0:
            x1 = 0
        if y1 < 0:
            y1 = 0
        if x2 > scale.shape[0]:
            x2 = scale.shape[0]
        if y2 > scale.shape[1]:
            y2 = scale.shape[1]
        neighbourhood = scale[x1:x2, y1:y2]

        # calculate the gradient direction and magnitude for each pixel in the patch
        dx = cv.Sobel(neighbourhood, cv.CV_64F, 1, 0, ksize=3)
        dy = cv.Sobel(neighbourhood, cv.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(dx**2 + dy**2)
        direction = np.arctan2(dy, dx)

        histogram[
            ((direction * 180 / np.pi) / 10).astype(int)
        ] += magnitude  # converts radians to degrees and then into one of the 36 bins

        max = np.max(histogram)
        # return any peaks above 80% of the maximum value
        directions = [i * 10 for i, value in enumerate(histogram) if value > 0.8 * max]
        keypoints = [
            Keypoint(centre=Point(i, j), scale=scale, orientation=direction)
            for direction in directions
        ]

        return keypoints

    def describe_keypoints(self, keypoints: list[Keypoint]) -> list[DescribedKeypoint]:
        described_keypoints = []

        for keypoint in keypoints:
            descriptor = []
            x1 = keypoint.centre.x - 8
            y1 = keypoint.centre.y - 8
            x2 = keypoint.centre.x + 8
            y2 = keypoint.centre.y + 8
            if x1 < 0:
                x1 = 0
            if y1 < 0:
                y1 = 0
            if x2 > keypoint.scale.shape[0]:
                x2 = keypoint.scale.shape[0]
            if y2 > keypoint.scale.shape[1]:
                y2 = keypoint.scale.shape[1]

            patch = keypoint.scale[
                keypoint.centre.x - 8 : keypoint.centre.x + 8,
                keypoint.centre.y - 8 : keypoint.centre.y + 8,
            ]

            # sub_patches = []
            for i in range(0, 16, 4):
                for j in range(0, 16, 4):
                    if (patch[i : i + 4, j : j + 4].size != 0):
                        # sub_patches.append(patch[i : i + 4, j : j + 4])
                        sub_patch = patch[i : i + 4, j : j + 4]
                    else:
                        # sub_patches.append(np.zeros((4, 4)))
                        sub_patch = np.zeros((4, 4))

                    histogram = np.zeros(8)

                    # calculate the gradient direction for each pixel in the patch - TODO repeated code
                    dx = cv.Sobel(sub_patch, cv.CV_64F, 1, 0, ksize=3)
                    dy = cv.Sobel(sub_patch, cv.CV_64F, 0, 1, ksize=3)

                    direction = np.arctan2(dy, dx)

                    histogram[
                        ((direction * 180 / np.pi) / 45).astype(int)
                    ] += 1  # sort into one of the 8 bins

                    descriptor.extend(histogram)

            descriptor = np.array(descriptor)
            # normalise for rotation
            descriptor -= keypoint.orientation
            # normalise for illumination
            max = np.max(histogram)
            descriptor *= 1 / max

            described_keypoints.append(
                DescribedKeypoint(keypoint=keypoint, descriptor=descriptor)
            )

        return described_keypoints

    def match_features(self, other_image_features: list[DescribedKeypoint]):
        matches = []
        for feature in self.features:
            feature_matches = []
            for feature_idx, other_feature in enumerate(other_image_features):
                # compare the descriptors using SSD
                SSD = np.sum((feature.descriptor - other_feature.descriptor) ** 2)
                if SSD < 0.8:
                    feature_matches.append((feature_idx, SSD))

            # sort the matches by SSD
            feature_matches.sort(key=lambda x: x[1])

            # if the best match is less than 80% of the second best match, add the match
            if feature_matches[0][1] < 0.8 * feature_matches[1][1]:
                matches.append((feature, other_image_features[feature_matches[0][0]]))

        return matches

if __name__ == "__main__":
    image_path = "/IconDataset/png/01-lighthouse.png"
    image = cv.imread(str(image_path))
    sift = Sift(image=image)
