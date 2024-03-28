from dataclasses import dataclass
from natsort import natsorted
from pathlib import Path
from scipy import ndimage
from typing import Iterator, List
from tqdm import tqdm

import cv2 as cv
import numpy as np
import os
import pandas as pd
import math

from gaussian_pyramid import GaussianPyramid


def print(s= ""):
    tqdm.write(str(s))


@dataclass
class Icon:
    top: int
    left: int
    bottom: int
    right: int
    label: str


@dataclass
class TestImage:
    name: str
    icons: List[Icon]


@dataclass
class Rectangle:
    x1: int
    y1: int
    x2: int
    y2: int

    def __mul__(self, other: float):
        return Rectangle(
            int(self.x1 * other),
            int(self.y1 * other),
            int(self.x2 * other),
            int(self.y2 * other),
        )


@dataclass
class Match:
    label: str
    difference: float
    bbox: Rectangle | None


def task2(folderName: str):
    version = 2

    this_file = Path(__file__)
    datasets_folder = this_file.parent.parent.parent / "datasets"
    dataset_folder = datasets_folder / folderName

    images_path = dataset_folder / "images"
    annotations_path = dataset_folder / "annotations"
    icon_dataset_path = datasets_folder / "IconDataset" / "png"
    # test_images = []

    # Load test results
    # for file in natsorted(os.listdir(annotations_path)):

    # Preprocess templates for matching
    icons: list[tuple[str, GaussianPyramid]] = []
    for file in os.listdir(icon_dataset_path):
        image_path = icon_dataset_path / file
        image = cv.imread(str(image_path))

        # replace all white/transparent pixels with black
        transparent_pixels = np.all(image == [255, 255, 255], axis=-1)
        image[transparent_pixels] = [0, 0, 0]

        # Create scaled templates
        # templates = create_scaled_templates(image, 7)
        pyramid = GaussianPyramid(image)

        # Create rotated templates
        # templates.extend(create_rotated_templates(templates))

        label = image_path.stem
        icon = (label, pyramid)
        icons.append(icon)

    # found_matches = []
    # Load test images and find matching icons
    for file in tqdm(natsorted(os.listdir(images_path))):
        # Load test image
        image = cv.imread(str(images_path / file))

        matches = find_matching_icons(image, icons, version)
        # found_matches.append(matches)

        icons_in_image = {}
        csv_file = (annotations_path / file).with_suffix(".csv")
        annotations = pd.read_csv(csv_file)

        for label, top, left, bottom, right in annotations.values:
            icons_in_image[label] = Icon(top, left, bottom, right, label)

        print()
        print(f"Test image: {file}")
        print(f"Found matches:")
        for match in matches:
            print(f"\t{match}")
        print(f"Icons in image: {list(icons_in_image.values())}")

        labels_in_image = [icon.label for icon in icons_in_image.values()]
        correct_matches = [match.label for match in matches if match.label in labels_in_image]
        accuracy = len(correct_matches) / len(labels_in_image)
        print(f"Accuracy: {accuracy * 100}%")

    # accuracies = []
    # for i, image in enumerate(test_images):
    # actual_icons = [icon.label for icon in image.icons]
    # found_matches_labels = [match.label for match in found_matches[i]]
    # accuracy = len(set(actual_icons) & set(found_matches_labels)) / len(actual_icons)
    # accuracies.append(accuracy)
    # print(f"Test image: {image.name}" + ", Accuracy: " + str(accuracy * 100) + "%")
    # print(f"Actual icons: {actual_icons}")
    # print(f"Found matches: {found_matches_labels}")

    # print()
    # print(f"Average accuracy: {np.average(accuracies) * 100}%")


def find_matching_icons_3a(image, icons) -> List[Match]:
    # image = cv.resize(image, (512, 512), interpolation=cv.INTER_LINEAR)

    # Get bounding boxes
    grayscale_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    bounding_boxes = get_bounding_boxes(grayscale_image)
    icon_size = 512
    matches = []

    for x, y, w, h in bounding_boxes:
        min_difference = math.inf
        best_template = ""
        bbox_match = None
        bounding_box = image[y : y + h, x : x + w]
        cv.rectangle(image, (x, y), (x + w, y + h), (200, 0, 0), 2)

        if w < h:
            scale_factor = h / icon_size
        else:
            scale_factor = w / icon_size

        # slide icon over image
        for label, templates in icons:
            for template in templates:
                resized_template = cv.resize(
                    template,
                    (int(icon_size * scale_factor), int(icon_size * scale_factor)),
                    interpolation=cv.INTER_LINEAR,
                )

                if (
                    resized_template.shape[0] > bounding_box.shape[0]
                    or resized_template.shape[1] > bounding_box.shape[1]
                ):
                    # pad bounding_box to match template size
                    pad_height = resized_template.shape[0] - bounding_box.shape[0]
                    pad_width = resized_template.shape[1] - bounding_box.shape[1]
                    padded_bounding_box = np.pad(
                        bounding_box, ((0, pad_height), (0, pad_width), (0, 0)), mode="constant"
                    )

                    bounding_box = padded_bounding_box

                corr_matrix = match_template(bounding_box, resized_template)

                if corr_matrix is not None:
                    if corr_matrix.min() < min_difference:
                        min_difference = corr_matrix.min()
                        best_template = label
                        bbox = Rectangle(x, y, x + w, y + h)

        cv.putText(image, best_template, (x, y), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2, cv.LINE_AA)
        # print(best_template)
        matches.append(Match(Path(best_template).stem, min_difference, bbox_match))
    # cv.imshow("image", image)
    # cv.waitKey(0)

    return matches


def find_matching_icons_1(image, icons) -> List[Match]:
    image = cv.resize(image, (64, 64), interpolation=cv.INTER_LINEAR)
    matches = []

    for label, templates in icons:
        min_difference = math.inf
        best_template = ""
        bbox_match = None
        for template in templates:
            corr_matrix = match_template(image, template)

            if corr_matrix is not None:
                if corr_matrix.min() < min_difference:
                    min_difference = corr_matrix.min()
                    best_template = label
                    x1 = corr_matrix.argmin(axis=0)[0]
                    x2 = corr_matrix.argmin(axis=1)[0]
                    bbox_match = Rectangle(x1, x2, x1 + template.shape[0], x2 + template.shape[1])
        matches.append(Match(Path(label).stem, min_difference, bbox_match))

    matches = sorted(matches, key=lambda x: x.min_difference)
    # print()
    # print(filter_matches(matches, 100))
    # cv.imshow("item", image)
    # cv.waitKey(0)

    return filter_matches(matches, 0.05)


def render(image: np.ndarray, bbox: Rectangle | None = None):
    if bbox is not None:
        image = image.copy()
        cv.rectangle(image, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), (200, 0, 0), 2)

    cv.imshow("image", image)
    cv.waitKey(0)


def find_matching_icons_2(image, icons: list[tuple[str, GaussianPyramid]]) -> list[Match]:
    # render(image)
    image_pyramid = GaussianPyramid(image)
    matches: list[Match] = []

    # for every icon
    for label, pyramid in tqdm(icons):
        # render(pyramid.image)
        correct_matches = [
            "37-post-office",
            "06-church",
            "45-museum",
            "35-police",
            "50-cemetery",
        ]
        # will_match = label.split(".")[0] in correct_matches
        # if not will_match:
        #     continue
        # print(label)
        # if label != "02-bike":
            # continue

        # the bounding box of the icon
        # (between pyramid levels, will need to be x2 for next levl)
        previous_layer_bbox = None

        # "confidence" in this icon being the one
        min_difference = np.inf

        # what scale factors to try at the next level
        # 0.1-1.0 at the lowest level, then will be narrowed down
        scale_factors_to_try: list[float] = np.linspace(0.1, 1.0, 9).tolist()

        # for every pyramid level
        for level_number in reversed(range(len(pyramid))):
            image_level = image_pyramid[level_number]
            template_level = pyramid[level_number]

            best_scale_factor = 0
            best_scale_factor_score = np.inf
            best_scale_factor_bbox = None

            # for every scale factor in the pyramid level
            for scale_factor in scale_factors_to_try:
                if scale_factor < 0.05:
                    continue

                scaled_template = template_level.scaled(scale_factor)
                scaled_image = image_level.image

                if previous_layer_bbox is not None:
                    # crop image to bounding box
                    MARGIN = 10
                    y1 = previous_layer_bbox.y1 * 2 - MARGIN
                    y2 = previous_layer_bbox.y2 * 2 + MARGIN
                    x1 = previous_layer_bbox.x1 * 2 - MARGIN
                    x2 = previous_layer_bbox.x2 * 2 + MARGIN
                    if 0 < x1 < x2 < scaled_image.shape[1] and 0 < y1 < y2 < scaled_image.shape[0]:
                        scaled_image = scaled_image[y1:y2, x1:x2]

                corr_matrix = match_template(scaled_image, scaled_template)

                min_difference = corr_matrix.min()
                if min_difference > 0.3:
                    # tmp: if we skipped a match
                    if label in correct_matches:
                        print(f"skipping {label} at level {level_number} ({min_difference=})")
                    continue  # no match

                if min_difference > best_scale_factor_score:
                    continue  # we already found a better scale factor

                # we found the best match so far of all the scale factors.
                # calculate the bounding box coords in the whole image
                # (the image might be cropped from the previous layer's bbox)

                x1, y1 = np.unravel_index(corr_matrix.argmin(), corr_matrix.shape)
                x1, y1 = int(x1), int(y1)  # because np.unravel_index() returns intp (not int)
                template_shape = scaled_template.shape

                bbox = Rectangle(x1, y1, x1 + template_shape[0], y1 + template_shape[1])

                if previous_layer_bbox is not None:
                    bbox.x1 += previous_layer_bbox.x1
                    bbox.x2 += previous_layer_bbox.x1
                    bbox.y1 += previous_layer_bbox.y1
                    bbox.y2 += previous_layer_bbox.y1

                best_scale_factor = scale_factor
                best_scale_factor_score = min_difference
                best_scale_factor_bbox = bbox

                # render image with bounding box
                # render(scaled_image, best_bbox)

            scale_factors_to_try = [
                best_scale_factor * 0.7,
                best_scale_factor * 0.85,
                best_scale_factor,
            ]
            previous_layer_bbox = best_scale_factor_bbox

        # else:  # didn't break
        # if found_match:
        match = Match(Path(label).stem, min_difference, previous_layer_bbox)
        matches.append(match)

        # print(repr(match))
        # render(image, previous_layer_bbox)

    matches.sort(key=lambda x: x.difference)
    return matches[:5]


def find_matching_icons(image, icons, version) -> list[Match]:
    match version:
        case 1:
            return find_matching_icons_1(image, icons)
        case 2:
            return find_matching_icons_2(image, icons)
        case 3:
            return find_matching_icons_3a(image, icons)
        case _:
            raise ValueError("Invalid version number")


def filter_matches(matches: List[Match], threshold: float) -> List[Match]:
    return [match for match in matches if match.difference < threshold]


def match_template(image, template):
    # get the dimensions of the image and the template
    image_height, image_width, _ = image.shape
    template_height, template_width, _ = template.shape

    if template_height > image_height or template_width > image_width:
        raise ValueError("Template dimensions exceed image dimensions")

    # create a result matrix to store the correlation values
    result = np.ones((image_height - template_height + 1, image_width - template_width + 1))

    # iterate through the image and calculate the correlation
    for y in range(image_height - template_height + 1):
        for x in range(image_width - template_width + 1):

            patch1 = image[y : y + template_height, x : x + template_width]
            if (patch1 == 255).all():
                continue

            # bbox = Rectangle(x, y, x + template_width, y + template_height)
            # render(image, bbox)

            result[x, y] = calculate_patch_similarity(
                patch1,
                template,
                True,
                False,
            )

    # render(result)
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
        if w > h:
            bounding_boxes.append((x, y, w, w))
        else:
            bounding_boxes.append((x, y, h, h))

    bounding_boxes = filter_bboxes_within(bounding_boxes)
    return bounding_boxes


def filter_bboxes_within(bounding_boxes):
    bboxes_to_filter = []
    for i, bbox1 in enumerate(bounding_boxes):
        for j, bbox2 in enumerate(bounding_boxes):
            if i != j:
                x1, y1, w1, h1 = bbox1
                x2, y2, w2, h2 = bbox2
                if x1 >= x2 and y1 >= y2 and x1 < x2 + w2 and y1 < y2 + h2:
                    if (w2 * h2) > (w1 * h1):
                        bboxes_to_filter.append(bbox1)
    filtered_bboxes = [bbox for bbox in bounding_boxes if bbox not in bboxes_to_filter]
    return filtered_bboxes


def create_scaled_templates(image, num_scales=5):
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
        diff = (patch1 - patch2).astype(np.int32) ** 2

        def imshow(image, name: str):
            pass
            image = cv.resize(image, (512, 512), interpolation=cv.INTER_NEAREST)
            cv.imshow(name, image)
            cv.waitKey(0)

        # imshow(patch1, "patch1")
        # imshow(patch2, "patch2")
        # imshow(diff, "diff")
        # cv.waitKey(0)

        d = diff.mean() / (255 ** 2)
        # print(d)
        # if d < 0.25:
        #     imshow(patch1, "patch1")
        #     imshow(patch2, "patch2")
        #     imshow(diff, "diff")
        #     cv.waitKey(0)
        return d

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
