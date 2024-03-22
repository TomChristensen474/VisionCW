from pathlib import Path
from dataclasses import dataclass
from scipy import ndimage
from typing import Iterator

import cv2 as cv
import numpy as np
import os
import pandas as pd
import math


def task2(folderName: str):
    list_file = Path(folderName) / "list.txt"
    icon_dataset_path = Path("IconDataset/png")

    # Preprocess templates for matching
    icons = []
    for file in os.listdir(icon_dataset_path):
        image = cv.imread(os.path.join(icon_dataset_path, file))

        # Create scaled templates
        templates = create_scaled_templates(image)

        # # Create rotated templates
        # templates = create_rotated_templates(templates)

        icon = [file, templates]
        icons.append(icon)

    # Load test images
    test_images = Path(folderName) / "images"
    for file in os.listdir(test_images):
        print(file)

        # Load test image
        image = cv.imread(os.path.join(test_images, file))

        find_matching_icon(image, icons)

        break


def find_matching_icon(image, icons) -> str:
    icon_size = 512

    image = cv.resize(image, (64, 64), interpolation=cv.INTER_LINEAR)
    grayscale_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    cv.imshow("image", image)
    cv.waitKey(0)

    # Get bounding boxes
    bounding_boxes = get_bounding_boxes(grayscale_image)

    for x, y, w, h in bounding_boxes:
        min_difference = math.inf
        best_template = ""
        item = image[y : y + h, x : x + w]
        cv.rectangle(image, (x, y), (x + w, y + h), (200, 0, 0), 2)

        bounding_box_size = (w, h)
        scale_factor = w / icon_size
        # cv.putText(
        #     image, icon_name, (x - 10, y - 10), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2, cv.LINE_AA
        # )

        # slide icon over image
        for icon_name, templates in icons:
            for template in templates:
                resized_template = cv.resize(
                    template,
                    (int(icon_size * scale_factor), int(icon_size * scale_factor)),
                    interpolation=cv.INTER_LINEAR,
                )

                corr_matrix = match_template(image, resized_template)

                if corr_matrix.min() < min_difference:
                    min_difference = corr_matrix.min()
                    best_template = icon_name
            
        print(best_template, min_difference)

    cv.imshow("item", image)
    cv.waitKey(0)

    #     matching_score_map[image_patch.centre_x][image_patch.centre_y] = similarity_score
    best_template = ""
    return best_template


def match_template(image, template):
    # get the dimensions of the image and the template
    image_height, image_width, _ = image.shape
    template_height, template_width, _ = template.shape

    # create a result matrix to store the correlation values
    result = np.zeros((image_height - template_height + 1, image_width - template_width + 1))

    # iterate through the image and calculate the correlation
    for y in range(image_height - template_height + 1):
        for x in range(image_width - template_width + 1):
            result[x, y] = calculate_patch_similarity(
                image[y : y + template_height, x : x + template_width], template, True, False
            )

    return result


def get_bounding_boxes(grayscale_image):
    # Threshold image
    _, threshold = cv.threshold(grayscale_image, 240, 255, cv.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv.findContours(threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Get bounding boxes
    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        # print(x, y, w, h)
        bounding_boxes.append((x, y, w, h))
        cv.rectangle(grayscale_image, (x, y), (x + w, y + h), (200, 0, 0), 2)

    return bounding_boxes


def create_scaled_templates(image, num_scales=3):
    scaled_templates = []

    for i in range(num_scales):
        scaled_templates.append(downsample(image, 0.5**i))

    return scaled_templates


def create_rotated_templates(templates, num_rotations=4):
    rotated_templates = []

    for template in templates:
        for i in range(num_rotations):
            rotated_templates.append(rotate_image(template, 90 * i))

    return rotated_templates


def rotate_image(image, rotation):
    return ndimage.rotate(image, rotation)


def downsample(image, scale_factor=0.5):
    downsampled_size = int(image.shape[0] * scale_factor), int(image.shape[1] * scale_factor)
    # gaussian blur image
    image = cv.GaussianBlur(image, (5, 5), 0)
    # scale down image
    resized_image = cv.resize(image, downsampled_size, interpolation=cv.INTER_LINEAR)

    return resized_image


def calculate_patch_similarity(
    patch1, patch2, ssd_match: bool = True, cross_corr_match: bool = False
) -> float:
    def ssd_normalized(patch1, patch2):
        # return cv.matchTemplate(patch1, patch2, cv.TM_SQDIFF_NORMED)
        # st_dev_patch1 = cv.meanStdDev(patch1)
        # st_dev_patch2 = cv.meanStdDev(patch2)

        # mean_patch1 = cv.mean(patch1)
        # mean_patch2 = cv.mean(patch2)

        # norm_patch1 = (patch1 - mean_patch1) / st_dev_patch1
        # norm_patch2 = (patch2 - mean_patch2) / st_dev_patch2

        # norm_patch1 = cv.normalize(patch1, patch1, 0, 255, cv.NORM_MINMAX)
        # norm_patch2 = cv.normalize(patch2, patch2, 0, 255, cv.NORM_MINMAX)
        if np.std(patch1) != 0 :
            norm_patch1 = (patch1 - np.mean(patch1)) / np.std(patch1)
        else:
            norm_patch1 = (patch1 - np.mean(patch1))
        if np.std(patch2) != 0:
            norm_patch2 = (patch2 - np.mean(patch2)) / np.std(patch2)
        else:
            norm_patch2 = (patch2 - np.mean(patch2))
        return np.sum(np.square(norm_patch1 - norm_patch2))

    if ssd_match == cross_corr_match:
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


if __name__ == "__main__":
    task2("Task2Dataset")
