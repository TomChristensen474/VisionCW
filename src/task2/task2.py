import time
from dataclasses import dataclass, field
from joblib import Parallel, delayed
from natsort import natsorted
from pathlib import Path
from scipy import ndimage
from typing import List, Literal
from tqdm import tqdm

import cv2 as cv
import numpy as np
import os
import pandas as pd
import math

from gaussian_pyramid import GaussianPyramid

Image = np.ndarray


@dataclass
class Config:
    debug_level: int = 0
    pyramid_levels: int = 2
    scale_factors: int | Literal["variable"] = "variable"
    threshold: float | Literal["variable"] = "variable"
    metric: Literal["mcc", "ssd"] = "mcc"
    cv_bbox_method: bool = False
    filter_nested: bool = True

    def is_debug(self, level: int) -> bool:
        return self.debug_level >= level


# config = Config()


def print(s=""):
    tqdm.write(str(s))


class CsvWriter:
    def __init__(self):
        self.path = Path(__file__).parent / "results.csv"

        self.config_columns = list(Config.__dataclass_fields__.keys())
        self.config_columns.remove("debug_level")

        self.columns = [
            "when",
            "accuracy",
            "TPR",
            "FPR",
            "FNR",
            "avg_IoU",
            "avg_runtime",
        ] + self.config_columns

        if not self.path.exists():  # if csv file doesn't exist, create it and write row names (first line)
            self.file = open(self.path, "w")
            self.file.write(",".join(self.columns) + "\n")
        else:  # if csv file exists, make sure row names are what we are going to write
            with self.path.open("r") as f:
                expected_header = ",".join(self.columns).replace(" ", "")
                actual_header = f.readline().strip().replace(" ", "")
                assert (
                    expected_header == actual_header
                ), f"CSV file has wrong header\nexpecting: {expected_header}\ngot:       {actual_header}"

            self.file = open(self.path, "a")

    def add(self, accuracy, tpr, fpr, fnr, avg_iou, avg_runtime):
        when = time.strftime("%Y-%m-%d %H:%M:%S")
        fields_to_write = [when, accuracy, tpr, fpr, fnr, avg_iou, avg_runtime]
        fields_to_write += [getattr(config, field) for field in self.config_columns]

        fields_to_write = [f"{x:.2f}" if isinstance(x, float) else str(x) for x in fields_to_write]
        self.file.write(",".join(fields_to_write) + "\n")


@dataclass
class Icon:
    label: str
    bbox: "Rectangle"


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

    def expanded(self, margin_percent: float, img_dimensions: tuple[int, int]) -> "Rectangle":
        margin = int(img_dimensions[0] * margin_percent)

        max_x, max_y = img_dimensions
        max_x2, max_y2 = max_x - margin, max_y - margin

        neg_x, pos_x = margin, margin
        neg_y, pos_y = margin, margin

        if self.x1 < margin:
            neg_x = self.x1
            pos_x += margin - self.x1
        elif self.x2 > max_x2:
            pos_x = max_x - self.x2
            neg_x += margin - (max_x - self.x2)

        if self.y1 < margin:
            neg_y = self.y1
            pos_y += margin - self.y1
        elif self.y2 > max_y2:
            pos_y = max_y - self.y2
            neg_y += margin - (max_y - self.y2)

        expanded = Rectangle(
            int(self.x1 - neg_x),
            int(self.y1 - neg_y),
            int(self.x2 + pos_x),
            int(self.y2 + pos_y),
        )

        # assert 0 <= x1 < x2 <= scaled_image.shape[1]
        assert 0 <= expanded.x1 < expanded.x2 <= max_x
        assert 0 <= expanded.y1 < expanded.y2 <= max_y
        return expanded

    def overlaps_with(self, other: "Rectangle") -> bool:
        # use separating axis theorem: https://stackoverflow.com/a/40795835/6087491
        return not (other.x2 <= self.x1 or self.x2 <= other.x1 or other.y2 <= self.y1 or self.y2 <= other.y1)

    def area(self) -> int:
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    def intersection(self, other: "Rectangle") -> "Rectangle | None":
        if not self.overlaps_with(other):
            return None

        return Rectangle(
            max(self.x1, other.x1),
            max(self.y1, other.y1),
            min(self.x2, other.x2),
            min(self.y2, other.y2),
        )

    def intersection_area(self, other: "Rectangle") -> int:
        intersection = self.intersection(other)
        return intersection.area() if intersection else 0

    def union_area(self, other: "Rectangle") -> int:
        return self.area() + other.area() - self.intersection_area(other)

    def __mul__(self, other: float):
        return Rectangle(
            int(self.x1 * other),
            int(self.y1 * other),
            int(self.x2 * other),
            int(self.y2 * other),
        )

    def __hash__(self) -> int:
        return hash((self.x1, self.y1, self.x2, self.y2))


@dataclass
class RatioRectangle:
    # to preserve bounding boxes between pyramid levels
    # all between 0 and 1
    x1: float
    y1: float
    x2: float
    y2: float

    @staticmethod
    def from_bbox(bbox: Rectangle, img_dimensions: tuple[int, int]):
        img_width, img_height = img_dimensions
        return RatioRectangle(
            bbox.x1 / img_width,
            bbox.y1 / img_height,
            bbox.x2 / img_width,
            bbox.y2 / img_height,
        )

    def to_absolute(self, img_dimensions: tuple[int, int], margin_percent: float = 0):
        img_width, img_height = img_dimensions

        x1 = max(round(self.x1 * img_width), 0)
        y1 = max(round(self.y1 * img_height), 0)
        x2 = min(round(self.x2 * img_width), img_width)
        y2 = min(round(self.y2 * img_height), img_height)

        rect = Rectangle(x1, y1, x2, y2)
        return rect.expanded(margin_percent, img_dimensions)

    def __mul__(self, other: float):
        return RatioRectangle(
            self.x1 * other,
            self.y1 * other,
            self.x2 * other,
            self.y2 * other,
        )


@dataclass
class Match:
    label: str
    difference: float
    bbox: Rectangle
    icon_image: Image = field(repr=False)

    def __hash__(self) -> int:  # to allow use in set()
        return hash((self.label, self.difference, self.bbox))


def task2(icon_folder_name: str, test_folder_name: str) -> tuple[float, float, float, float]:
    this_file = Path(__file__)
    datasets_folder = this_file.parent.parent.parent / "datasets"
    dataset_folder = datasets_folder / test_folder_name

    images_path = dataset_folder / "images"
    annotations_path = dataset_folder / "annotations"
    icon_dataset_path = datasets_folder / icon_folder_name / "png"

    totals = {
        "positives": 0,
        "negatives": 0,
        "true_positives": 0,
        "true_negatives": 0,
        "false_positives": 0,
        "false_negatives": 0,
    }
    runtimes = []
    all_ious = []

    # Preprocess templates for matching
    icons: list[tuple[str, GaussianPyramid]] = []
    for icon_file in os.listdir(icon_dataset_path):
        icon_path = icon_dataset_path / icon_file
        icon_image = cv.imread(str(icon_path))

        # Create scaled templates
        icon_pyramid = GaussianPyramid(icon_image, config.pyramid_levels)

        # Create rotated templates
        # templates.extend(create_rotated_templates(templates))

        icon_label = icon_path.stem
        icon = (icon_label, icon_pyramid)
        icons.append(icon)

    # found_matches = []
    # Load test images and find matching icons
    for file in tqdm(natsorted(os.listdir(images_path)), desc="test image"):
        # Load test image
        image = cv.imread(str(images_path / file))

        start_time = time.time()
        matches = find_matching_icons(image, icons)
        end_time = time.time()

        runtimes.append(end_time - start_time)

        icons_in_image: dict[str, Icon] = {}
        csv_file = (annotations_path / file).with_suffix(".csv")
        annotations = pd.read_csv(csv_file)

        for label, left, top, right, bottom in annotations.values:
            bbox = Rectangle(left, top, right, bottom)
            icons_in_image[label] = Icon(label, bbox)

        print(f"\nTest image: {file}")

        # == calculate all the metrics ==

        all_labels = set(label for (label, _) in icons)
        matched_labels = set(match.label for match in matches)

        # calculate intersection over union (IoU)
        ious = {}
        for icon in icons_in_image.values():
            # get the corresponding match
            match = next((m for m in matches if m.label == icon.label), None)
            if match is None:
                ious[icon.label] = 0
                continue

            correct_bbox = icon.bbox
            predicted_bbox = match.bbox

            intersection = correct_bbox.intersection_area(predicted_bbox)
            union = correct_bbox.union_area(predicted_bbox)

            iou = intersection / union
            ious[icon.label] = iou
            all_ious.append(iou)

            # if the IoU is <50%, it's not a match
            # => remove it from the list of matches
            if iou < 0.5:
                matched_labels.remove(icon.label)

        positives = set(icon.label for icon in icons_in_image.values())  # labels in image
        negatives = all_labels - positives  # labels not in image

        true_positives = matched_labels & positives
        false_positives = matched_labels - positives
        false_negatives = positives - matched_labels
        true_negatives = all_labels - matched_labels - positives

        totals["positives"] += len(positives)
        totals["negatives"] += len(negatives)
        totals["true_positives"] += len(true_positives)
        totals["true_negatives"] += len(true_negatives)
        totals["false_positives"] += len(false_positives)
        totals["false_negatives"] += len(false_negatives)

        tpr = len(true_positives) / len(positives)  # i.e. recall
        fpr = len(false_positives) / len(negatives)
        fnr = len(false_negatives) / len(positives)
        accuracy = (len(true_positives) + len(true_negatives)) / len(all_labels)

        print(f"Accuracy: {accuracy * 100:.3}% TPR: {tpr * 100:.3}% FPR: {fpr * 100:.3}% FNR: {fnr * 100:.3}%")

        print("True positives: (correct matches)")
        for label in true_positives:
            print(f"\t{label:<10} IoU: {ious[label]:.3f}")

        print("False positives:")
        for label in false_positives:
            print(f"\t{label:<10}")

        print("False negatives: (missed matches)")
        for label in false_negatives:
            print(f"\t{label:<10}")

        print("\n")

    total_accuracy = (totals["true_positives"] + totals["true_negatives"]) / (
        totals["positives"] + totals["negatives"]
    )
    total_tpr = totals["true_positives"] / totals["positives"]
    total_fpr = totals["false_positives"] / totals["negatives"]
    total_fnr = totals["false_negatives"] / totals["positives"]

    print("\n Total final stats:")
    print(str(totals))
    print(
        f"Accuracy: {total_accuracy * 100:.2f}% TPR: {total_tpr * 100:.2f}% FPR: {total_fpr * 100:.2f}% FNR: {total_fnr * 100:.2f}%"
    )

    average_iou = sum(all_ious) / len(all_ious)
    print(f"Average IoU: {average_iou * 100:.2f}%")

    average_runtime = sum(runtimes) / len(runtimes)
    print(f"Average runtime: {average_runtime:.2f}s")

    csv_writer = CsvWriter()
    csv_writer.add(
        accuracy=total_accuracy,
        tpr=total_tpr,
        fpr=total_fpr,
        fnr=total_fnr,
        avg_iou=average_iou,
        avg_runtime=average_runtime,
    )

    return (total_accuracy, total_tpr, total_fpr, total_fnr)


def find_matching_icons_3a(image: Image, icons: list[tuple[str, GaussianPyramid]]) -> List[Match]:

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

        scale_factor = max(w, h) / icon_size

        # slide icon over image
        for label, pyramid in icons:
            for pyramid_level in pyramid:
                template = pyramid_level.image
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
                        bounding_box,
                        ((pad_height // 2, pad_height // 2), (pad_width // 2, pad_width // 2), (0, 0)),
                        mode="constant",
                    )

                    bounding_box = padded_bounding_box

                corr_matrix = match_template(bounding_box, resized_template)

                if corr_matrix is not None and corr_matrix.min() < min_difference:
                    min_difference = corr_matrix.min()
                    best_template = label
                    bbox_match = Rectangle(x, y, x + w, y + h)

        assert bbox_match is not None
        label = Path(best_template).stem
        icon_image = next(pyramid.image for (_label, pyramid) in icons if label == _label)
        match = Match(label, min_difference, bbox_match, icon_image)
        matches.append(match)

    matches = filter_nested_matches(matches)

    render(image, matches, wait=False)
    return matches


def render(
    img: Image,
    matches: List[Match] = [],
    window_name="image",
    wait=False,  # wait for keypress
):
    image = img.copy()
    icons = []
    search_areas = []

    # draw labelled bounding boxes on image
    for match in matches:
        # draw bbox
        bbox = match.bbox
        assert bbox is not None
        cv.rectangle(image, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), (200, 0, 0), 2)

        # draw label text
        label_text = f"{match.label} {match.difference:.3f}"
        cv.putText(
            image,
            label_text,
            (bbox.x1, bbox.y2 + 10),
            cv.FONT_HERSHEY_PLAIN,
            1,
            (0, 0, 255),
            # 2,
            # cv.LINE_AA,
        )

        # add icon
        icons.append(match.icon_image)

        # add search area
        search_area = img[bbox.y1 : bbox.y2, bbox.x1 : bbox.x2]
        search_areas.append(search_area)

    assert len(icons) == len(search_areas) == len(matches)

    if len(matches) > 1:
        # multi-match layout
        # image is on the left, at its resolution of 512x512.
        # icons and search areas are scaled to 1/2 its side length, one above the other
        # and are placed to the right of the image

        match_width = int(image.shape[1] / 2)
        match_height = int(image.shape[0] / 2)

        icons = [cv.resize(icon, (match_width, match_height), interpolation=cv.INTER_NEAREST) for icon in icons]
        icons = np.hstack(icons)

        search_areas = [
            cv.resize(search_area, (match_width, match_height), interpolation=cv.INTER_NEAREST)
            for search_area in search_areas
        ]
        search_areas = np.hstack(search_areas)

        everything_except_image = np.vstack((icons, search_areas))
        image = np.hstack((image, everything_except_image))

    elif len(matches) == 1:
        # single match layout
        # just resize and hstack the image, search area and icon

        icon = cv.resize(icons[0], (image.shape[1], image.shape[0]), interpolation=cv.INTER_NEAREST)
        search_area = cv.resize(search_areas[0], (image.shape[1], image.shape[0]), interpolation=cv.INTER_NEAREST)

        image = np.hstack((image, search_area, icon))

    cv.imshow(window_name, image)
    cv.waitKey(0 if wait else 1)


def find_matching_icons_2(image, icons: list[tuple[str, GaussianPyramid]]) -> list[Match]:
    @delayed
    def get_label_icon_match(image, icon_label_and_pyramid) -> Match | None:
        icon_label, icon_pyramid = icon_label_and_pyramid
        image_pyramid = GaussianPyramid(image, config.pyramid_levels)

        if config.scale_factors == "variable":
            scale_factor_multipliers_per_level = [
                [1.0],
                np.linspace(0.95, 1.05, 3).tolist(),
                np.linspace(0.8, 1.2, 5).tolist(),
                np.linspace(0.1, 0.9, 8).tolist(),
                np.linspace(0.05, 0.95, 10).tolist(),
            ]
        else:
            scale_factor_multipliers_per_level = [
                [1.0],
                np.linspace(0.95, 1.05, 3 * config.scale_factors).tolist(),
                np.linspace(0.8, 1.2, 5 * config.scale_factors).tolist(),
                np.linspace(0.1, 0.9, 8 * config.scale_factors).tolist(),
                np.linspace(0.05, 0.95, 10 * config.scale_factors).tolist(),
            ]

        if config.threshold == "variable":
            thresholds_per_level = [0.15, 0.2, 0.3, 0.4]
        else:
            thresholds_per_level = [config.threshold] * config.pyramid_levels

        # the bounding box of the search space i.e. where the icon was on the previous level
        # (between pyramid levels, will need to be x2 for next level)
        layer_bbox: RatioRectangle | None = None

        # "confidence" in this icon being the one
        min_difference = np.inf

        # the scale factor of the previous pyramid layer
        # starts at 1 because this value will be multiplied by numbers like 0.9
        previous_layer_scale_factor = 1.0
        previous_layer_scale_factor_level = -1  # to know if we made it to the last level

        # for every pyramid level
        for level_number in reversed(range(len(icon_pyramid))):

            image_level = image_pyramid[level_number]
            template_level = icon_pyramid[level_number]
            scale_factor_multipliers = scale_factor_multipliers_per_level[level_number]

            best_scale_factor = 1.0
            best_scale_factor_score = np.inf
            best_scale_factor_bbox = None

            # for every scale factor in the pyramid level
            for scale_factor_multiplier in scale_factor_multipliers:
                scale_factor = scale_factor_multiplier * previous_layer_scale_factor

                if scale_factor < 0.05:
                    continue

                scaled_template = template_level.scaled(scale_factor)
                scaled_image = image_level.image

                if layer_bbox is not None:  # crop image to previous layers' bounding box

                    bbox = layer_bbox.to_absolute((scaled_image.shape[1], scaled_image.shape[0]), 0.05)
                    assert bbox.x2 - bbox.x1 == bbox.y2 - bbox.y1  # absolute bbox should be perfecly square

                    x1, x2 = bbox.x1, bbox.x2
                    y1, y2 = bbox.y1, bbox.y2

                    assert 0 <= x1 < x2 <= scaled_image.shape[1]
                    assert 0 <= y1 < y2 <= scaled_image.shape[0]
                    image_search_region = scaled_image[y1:y2, x1:x2]

                else:  # no cropping necessary, look through whole image
                    image_search_region = scaled_image
                    bbox = None

                # if template is bigger than image, skip
                # (this might happen if scale_factor is >1 several layers in a row)
                if (
                    scaled_template.shape[0] > image_search_region.shape[0]
                    or scaled_template.shape[1] > image_search_region.shape[1]
                ):
                    continue

                corr_matrix = match_template(image_search_region, scaled_template)

                x_min, y_min = np.unravel_index(corr_matrix.argmin(), corr_matrix.shape)
                min_difference = corr_matrix[x_min, y_min]
                threshold = thresholds_per_level[level_number]
                if min_difference > threshold:  # <- tunable hyperparameter
                    continue  # no match

                if min_difference > best_scale_factor_score:
                    continue  # we already found a better scale factor

                # we found the best match of all the scale factors so far.
                best_scale_factor = scale_factor
                best_scale_factor_score = min_difference

                # calculate the bounding box coords so that the next layer only
                # has to look here
                x1, y1 = int(x_min), int(y_min)  # because np.unravel_index() returns intp (not int)

                # the image might be cropped from the previous layer's bbox
                if bbox is not None:
                    x1 += bbox.x1
                    y1 += bbox.y1

                template_shape = scaled_template.shape
                best_scale_factor_bbox = Rectangle(x1, y1, x1 + template_shape[0], y1 + template_shape[1])

                if config.is_debug(1):
                    # render image with bounding box
                    match = Match(icon_label, min_difference, best_scale_factor_bbox, scaled_template)
                    render(scaled_image, [match])

            if best_scale_factor_score == np.inf:
                # all scales were skipped due to not being similar enough
                break

            previous_layer_scale_factor = best_scale_factor
            previous_layer_scale_factor_level = level_number

            assert best_scale_factor_bbox is not None
            img_dimensions = (image_level.image.shape[1], image_level.image.shape[0])
            layer_bbox = RatioRectangle.from_bbox(best_scale_factor_bbox, img_dimensions)

        if layer_bbox is None:
            return None  # this icon is not even close

        if previous_layer_scale_factor_level != 0:
            # we matched on lower levels, but we stopped matching before reaching the highest level
            return None

        bbox = layer_bbox.to_absolute((image.shape[1], image.shape[0]), 0)
        match = Match(Path(icon_label).stem, min_difference, bbox, icon_pyramid.image)

        return match

    matches_generator = Parallel(n_jobs=-1, return_as="generator")(
        get_label_icon_match(image, icon_label_and_pyramid) for icon_label_and_pyramid in icons
    )
    matches_generator = tqdm(matches_generator, total=len(icons), desc="Icons")
    matches = [match for match in matches_generator if match is not None]

    matches = filter_nested_matches(matches)

    # render all the matches
    render(image, matches, window_name="all matches")

    matches.sort(key=lambda x: x.difference)
    return matches


def find_matching_icons(image: Image, icons: list[tuple[str, GaussianPyramid]]) -> list[Match]:
    if config.cv_bbox_method:
        return find_matching_icons_3a(image, icons)
    else:
        return find_matching_icons_2(image, icons)


def filter_nested_matches(matches: list[Match]) -> list[Match]:
    if not config.filter_nested:
        return matches

    # detect nested bounding boxes, and only keep the largest one
    matches_to_remove = set()
    for match1 in matches:
        for match2 in matches:
            if match1 == match2:
                continue

            assert match1.bbox is not None
            assert match2.bbox is not None

            if match1.bbox.overlaps_with(match2.bbox):
                bbox1_sides = (match1.bbox.x2 - match1.bbox.x1) + (match1.bbox.y2 - match1.bbox.y1)
                bbox2_sides = (match2.bbox.x2 - match2.bbox.x1) + (match2.bbox.y2 - match2.bbox.y1)

                match_to_remove = match2 if bbox1_sides > bbox2_sides else match1
                matches_to_remove.add(match_to_remove)

    filtered_matches = [match for match in matches if match not in matches_to_remove]
    return filtered_matches


def match_template(image: Image, template: Image):
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
            if (patch1 == 255).all() or (patch1 == 0).all():
                continue

            if config.is_debug(2):
                bbox = Rectangle(x, y, x + template_width, y + template_height)
                match = Match("", 0, bbox, template)
                render(image, [match])

            result[x, y] = calculate_patch_similarity(
                patch1,
                template,
                config.metric == "ssd",
                config.metric == "mcc",
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
            padding = w - h
            y -= padding // 2
            y = max(0, y)
            h = w
        elif w < h:
            padding = h - w
            x -= padding // 2
            x = max(0, x)
            w = h

        bbox = (x, y, w, h)
        bounding_boxes.append(bbox)

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


def calculate_patch_similarity(patch1, patch2, ssd_match: bool = True, cross_corr_match: bool = False) -> float:
    def imshow(image, name: str):
        image = cv.resize(image, (512, 512), interpolation=cv.INTER_NEAREST)
        cv.imshow(name, image)
        cv.waitKey(0)

    def ssd_normalized(patch1, patch2):
        diff = (patch1 - patch2).astype(np.float32) ** 2
        normalized_diff = diff / (255**2)
        return normalized_diff.mean()

    def cross_corr_normalized(patch1, patch2):
        # convert to float
        patch1 = patch1.astype(np.float32)
        patch2 = patch2.astype(np.float32)

        # normalize
        def normalise_patch(patch):
            patch -= patch.min()
            patch *= 255 / patch.max()
            return patch

        patch1_norm = normalise_patch(patch1)
        patch2_norm = normalise_patch(patch2)

        patch1_hat = (patch1_norm - patch1_norm.mean()) / patch1_norm.std()
        patch2_hat = (patch2_norm - patch2_norm.mean()) / patch2_norm.std()

        diff = patch1_hat * patch2_hat

        if config.is_debug(2):
            # norm_diff = 1. - (diff + 1.) / 2. * 255.
            # imshow(norm_diff, "diff")
            pass

        mean_diff = diff.mean()  # [-1, 1]
        assert not np.isnan(mean_diff)

        mean_diff = (mean_diff + 1.0) / 2.0  # [0, 1]
        mean_diff = 1.0 - mean_diff  # [1, 0] because 0 is "no difference"

        return mean_diff

    if ssd_match == cross_corr_match:
        raise ValueError("Choose correlation matching or ssd matching!")

    match_score: float
    if ssd_match:
        match_score = ssd_normalized(patch1, patch2)
    elif cross_corr_match:
        match_score = cross_corr_normalized(patch1, patch2)
    else:
        raise ValueError("Choose correlation matching or ssd matching!")

    assert 0 <= match_score <= 1

    if config.is_debug(2):
        if match_score < 0.3:
            print(str(match_score))
            imshow(patch1, "patch1")
            imshow(patch2, "patch2")
            # imshow(match_score / 25565, "match_score")
            cv.waitKey(0)

    return match_score


if __name__ == "__main__":
    configs = [
        Config(pyramid_levels=3, scale_factors=1, threshold=0.15, metric="ssd"),
        Config(pyramid_levels=3, scale_factors=1, threshold=0.15, metric="mcc"),
        Config(pyramid_levels=4, scale_factors=1, threshold=0.15, metric="mcc"),
        Config(pyramid_levels=5, scale_factors=1, threshold=0.15, metric="mcc"),
        # Config(pyramid_levels=4, scale_factors=4, threshold=0.15, metric="mcc"),
        # Config(pyramid_levels=4, scale_factors=6, threshold=0.15, metric="mcc"),
    ]

    # # running multiple configs
    for config in configs:
        print("\n Running config: " + str(config))
        task2("IconDataset", "Task2Dataset")

    # # # running single config
    # config = Config()
    # task2("IconDataset", "Task2Dataset")
