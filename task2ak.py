import os
from pathlib import Path
from typing import List, Tuple
import cv2 as cv
from dataclasses import dataclass

import numpy as np

@dataclass
class ImagePatch:
    centre_x: int
    centre_y: int
    image_patch: List[List[float]]

def task2(folderName: str) -> float:
    # TODO: don't harcode paths
    path_dir = os.getcwd() + "/IconDataset/png/001-lighthouse.png"

    img = cv.imread(str(path_dir))
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    print(img)
    return 0.0

def extract_patches_from_image(input_image):
    # TODO: extract patches from image

    # PLACEHOLDER to avoid typing error
    p1 = ImagePatch(centre_x=1,centre_y=1,image_patch=[[1,2,3],[1,2,3]])
    image_patches: List[ImagePatch] = [p1]
    return image_patches

def calculate_patch_similarity(patch1, patch2) -> float:
    # TODO: use similarity score function (e.g. SDD / block matching)

    # TODO: add normalization of intensity
    return 0.0

def template_matching(input_image, template_image):
    IX = len(input_image)
    IY = len(input_image[0])

    TX = len(template_image)
    TY = len(template_image[0])

    image_patches = extract_patches_from_image(input_image=input_image)
    PATCH_MATCH_THRESHOLD = 1  # change this

    matching_score_map = np.empty([IX, IY], dtype = float)
    for image_patch in image_patches:
        similarity_score = calculate_patch_similarity(image_patch, template_image)
        if similarity_score < PATCH_MATCH_THRESHOLD:
            matching_score_map[image_patch.centre_x][image_patch.centre_y] = similarity_score

    return matching_score_map



if __name__ == "__main__":
    task2("Task2Dataset")
