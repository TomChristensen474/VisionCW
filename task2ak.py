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

def compute_max_location(matching_score_map):
    # TODO change this to coordinates of max locaiton
    x:int = 0
    y:int = 0
    return (x,y)

def calculate_patch_similarity(patch1, patch2, ssd_match:bool = True, cross_corr_match:bool = False) -> float:
    def ssd_normalized(patch1, patch2):
        norm_patch1 = patch1 - np.mean(patch1)
        norm_patch2 = patch2 - np.mean(patch2)
        return np.sum(np.square(norm_patch1 - norm_patch2))

    if (ssd_match == cross_corr_match):
        raise ValueError("Choose correlation matching or ssd matching!")

    match_score: float 
    if ssd_match:
        match_score = ssd_normalized(patch1=patch1, patch2=patch2)
    elif cross_corr_match:
        # TODO: add option for this
        pass
    else:
        raise ValueError("Choose correlation matching or ssd matching!")

    return match_score

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

    x,y = compute_max_location(matching_score_map=matching_score_map)

    return (x,y)

# TEMPLATE IMAGE = a single image in training
# we need
def run(input_image, template_image):
    max_match_point = template_matching(input_image, template_image)

if __name__ == "__main__":
    task2("Task2Dataset")
