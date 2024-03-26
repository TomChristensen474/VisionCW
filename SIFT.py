from dataclasses import dataclass

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

class SIFT:
    features: list[Feature] | None

    def __init__(self, image):
        self.image = image

    def scale_space_construct(self):
        pass

    def detect_keypoints(self):
        pass

    def describe_keypoints(self):
        pass

    def match_features(self, other_features: list[Feature]):
        pass