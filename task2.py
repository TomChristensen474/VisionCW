from pathlib import Path
from dataclasses import dataclass
from scipy import ndimage
from typing import Iterator

import cv2 as cv
import numpy as np
import os
import pandas as pd


def task2(folderName: str):
    list_file = Path(folderName) / "list.txt"
    icon_dataset_path = Path("IconDataset/png")

    # Preprocess templates for matching
    icons = []
    for file in os.listdir(icon_dataset_path):
        print(file)
        image = cv.imread(os.path.join(icon_dataset_path,file))

        # Create scaled templates
        templates = create_scaled_templates(image)

        # Create rotated templates
        templates = create_rotated_templates(templates)

        icon = [file, templates]
        icons.append(icon)
        # for template in templates:
        #     cv.imshow("rotated", template)
        #     cv.waitKey(0)
    print(len(icons))
    # Load test images
    test_images = Path(folderName) / "images"
    for file in os.listdir(test_images):
        print(file)

        # Load test image
        image = cv.imread(os.path.join(test_images, file))
        # cv.imshow("image", image)
        # cv.waitKey(0)
        grayscale_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        # Get bounding boxes
        bounding_boxes = get_bounding_boxes(grayscale_image)

        # split up the image into images with bounding boxes
        for x, y, w, h in bounding_boxes:
            item = image[y : y + h, x : x + w]
            cv.rectangle(image, (x, y), (x + w, y + h), (200, 0, 0), 2)
            
            icon_name = match_template(image, item, [x,y,w,h], icons)

            cv.putText(image, icon_name, (x-10, y-10), cv.FONT_HERSHEY_PLAIN , 1, (0, 0, 255), 2, cv.LINE_AA)
        cv.imshow("item", image)
        cv.waitKey(0)

def match_template(image, item, bounding_box, icons) -> str:
    max_correlation = -1
    best_template = ""

    for icon_name, templates in icons:
        for template in templates:
            correlation = cv.matchTemplate(item, template, cv.TM_CCOEFF_NORMED) # CREATE OUR OWN MATCH FUNCTION
            if correlation.max() > max_correlation:
                max_correlation = correlation.max()
                best_template = icon_name

    return best_template
        

def get_bounding_boxes(grayscale_image):
    # Threshold image
    _, threshold = cv.threshold(
        grayscale_image, 240, 255, cv.THRESH_BINARY_INV
    )

    # Show threshold for debugging
    # cv.imshow("threshold", threshold)
    # cv.waitKey(0)

    # Find contours
    contours, _ = cv.findContours(threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Get bounding boxes
    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        print(x, y, w, h)
        bounding_boxes.append((x, y, w, h))
        cv.rectangle(grayscale_image, (x, y), (x + w, y + h), (200, 0, 0), 2)

    # Show bounding boxes for debugging
    # cv.imshow("img", grayscale_image)
    # cv.waitKey(0)

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
    downsampled_size = int(image.shape[0] * scale_factor), int(
        image.shape[1] * scale_factor
    )
    # gaussian blur image
    image = cv.GaussianBlur(image, (5, 5), 0)
    # scale down image
    resized_image = cv.resize(image, downsampled_size, interpolation=cv.INTER_LINEAR)

    return resized_image


if __name__ == "__main__":
    task2("Task2Dataset")
