import dataclasses
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

Image = np.ndarray


DEBUG_LEVEL = 0


def print(s=""):
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
    # to preserve bounding boxes for different
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
    bbox: Rectangle | None

    def __hash__(self) -> int:  # to allow use in set()
        return hash((self.label, self.difference, self.bbox))


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
        # transparent_pixels = np.all(image == [255, 255, 255], axis=-1)
        # image[transparent_pixels] = [0, 0, 0]

        # downscale to 100*100
        # image = cv.resize(image, (100, 100))

        # gaussian blur
        # image = cv.GaussianBlur(image, (5, 5), 1)

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
    for file in tqdm(natsorted(os.listdir(images_path)), desc="test image"):
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
            correct_match = "YES:" if match.label in icons_in_image else "NO: "
            print(f"\t{correct_match} {match}")

        correct_icons_string = "\n\t".join([str(icon) for icon in icons_in_image.values()])
        print(f"Icons in image:\n\t{correct_icons_string}")

        labels_in_image = [icon.label for icon in icons_in_image.values()]
        correct_matches = [match.label for match in matches if match.label in labels_in_image]
        accuracy = len(correct_matches) / len(labels_in_image)
        print(f"Accuracy: {accuracy * 100}%")

        # break  # tmp: stop after 1st image

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
                        bounding_box,
                        ((0, pad_height), (0, pad_width), (0, 0)),
                        mode="constant",
                    )

                    bounding_box = padded_bounding_box

                corr_matrix = match_template(bounding_box, resized_template)

                if corr_matrix is not None:
                    if corr_matrix.min() < min_difference:
                        min_difference = corr_matrix.min()
                        best_template = label
                        bbox = Rectangle(x, y, x + w, y + h)

        cv.putText(
            image,
            best_template,
            (x, y),
            cv.FONT_HERSHEY_PLAIN,
            1,
            (0, 0, 255),
            2,
            cv.LINE_AA,
        )
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


def render(image: np.ndarray, bbox: Rectangle | None = None, icon: np.ndarray | None = None):
    images_to_show = [image]

    if bbox is not None:
        image_with_bbox = image.copy()
        cv.rectangle(image_with_bbox, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), (200, 0, 0), 2)
        images_to_show[0] = image_with_bbox

        search_area = image[bbox.y1 : bbox.y2, bbox.x1 : bbox.x2]
        scaled_search_area = cv.resize(
            search_area, (image.shape[1], image.shape[0]), interpolation=cv.INTER_NEAREST
        )
        images_to_show.append(scaled_search_area)

    if icon is not None:
        # rescale icon to same size as image
        scaled_icon = cv.resize(icon, (image.shape[1], image.shape[0]), interpolation=cv.INTER_NEAREST)
        images_to_show.append(scaled_icon)

    # hstack the images
    all_images = np.hstack(images_to_show)

    cv.imshow("image", all_images)
    cv.waitKey(1)


def find_matching_icons_2(image, icons: list[tuple[str, GaussianPyramid]]) -> list[Match]:
    # render(image)
    image_pyramid = GaussianPyramid(image)
    matches: list[Match] = []

    scale_factor_multipliers_per_level = [
        [1.0],
        np.linspace(0.95, 1.05, 3).tolist(),
        np.linspace(0.8, 1.2, 5).tolist(),
        np.linspace(0.1, 0.9, 8).tolist(),
    ]

    # for every icon
    # for label, pyramid in tqdm(icons, desc="icon"):
    for icon_label, icon_pyramid in icons:
        # render(pyramid.image)
        # correct_matches = [
        #     "37-post-office",
        #     "06-church",
        #     "45-museum",
        #     "35-police",
        #     "50-cemetery",
        # ]
        # will_match = label.split(".")[0] in correct_matches
        # if not will_match:
        #     continue
        # print(label)
        # if label != "02-bike":
        # continue

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
                if min_difference > 0.15:  # <- tunable hyperparameter
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

                if DEBUG_LEVEL >= 1:
                    # render image with bounding box
                    render(scaled_image, best_scale_factor_bbox, icon=scaled_template)

            if best_scale_factor_score == np.inf:
                # all scales were skipped due to not being similar enough
                break

            previous_layer_scale_factor = best_scale_factor
            previous_layer_scale_factor_level = level_number

            assert best_scale_factor_bbox is not None
            img_dimensions = (image_level.image.shape[1], image_level.image.shape[0])
            layer_bbox = RatioRectangle.from_bbox(best_scale_factor_bbox, img_dimensions)

        if layer_bbox is None:
            continue  # this icon is not even close

        if previous_layer_scale_factor_level != 0:
            # we matched on lower levels, but we stopped matching before reaching the highest level
            continue

        bbox = layer_bbox.to_absolute((image.shape[1], image.shape[0]), 0)
        match = Match(Path(icon_label).stem, min_difference, bbox)
        matches.append(match)

        print(repr(match))
        # render(image, bbox, icon_pyramid.image)

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

    matches = [match for match in matches if match not in matches_to_remove]

    # render all the matches
    img_result = image.copy()
    for match in matches:
        assert match.bbox is not None
        cv.rectangle(
            img_result,
            (match.bbox.x1, match.bbox.y1),
            (match.bbox.x2, match.bbox.y2),
            (255, 0, 0),
            2,
        )

    cv.imshow("all bboxes", img_result)
    cv.waitKey(1)

    matches.sort(key=lambda x: x.difference)
    # return matches[:5]
    return matches


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


def match_template(image: Image, template: Image):
    # get the dimensions of the image and the template
    image_height, image_width, _ = image.shape
    template_height, template_width, _ = template.shape

    if template_height > image_height or template_width > image_width:
        raise ValueError("Template dimensions exceed image dimensions")

    # create a result matrix to store the correlation values
    result = np.ones((image_height - template_height + 1, image_width - template_width + 1))

    # iterate through the image and calculate the correlation
    # for y in tqdm(range(image_height - template_height + 1), desc="match_template y"):
    for y in range(image_height - template_height + 1):
        for x in range(image_width - template_width + 1):

            patch1 = image[y : y + template_height, x : x + template_width]
            if (patch1 == 255).all():
                continue

            if DEBUG_LEVEL >= 2:
                bbox = Rectangle(x, y, x + template_width, y + template_height)
                render(image, bbox, icon=template)

            result[x, y] = calculate_patch_similarity(
                patch1,
                template,
                False,
                True,
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
    def imshow(image, name: str):
        image = cv.resize(image, (512, 512), interpolation=cv.INTER_NEAREST)
        cv.imshow(name, image)
        cv.waitKey(0)

    def ssd_normalized(patch1, patch2):
        diff = (patch1 - patch2).astype(np.int32) ** 2
        return diff

    def cross_corr_normalized(patch1, patch2):
        # convert to float
        patch1 = patch1.astype(np.float32)
        patch2 = patch2.astype(np.float32)

        # normalize

        patch1 -= patch1.min()
        patch1 *= 255 / patch1.max()
        patch2 -= patch2.min()
        patch2 *= 255 / patch2.max()

        diff = ((patch1 - patch1.mean()) / patch1.std()) * ((patch2 - patch2.mean()) / patch2.std())

        if DEBUG_LEVEL >= 2:
            # norm_diff = 1. - (diff + 1.) / 2. * 255.
            # imshow(norm_diff, "diff")
            pass

        diff = np.sum(diff) / patch1.size

        # [-1, 1] -> [0, 1]
        diff = (diff + 1.0) / 2.0

        # flipping 0<x<1 to 1<x<0 because 0 is "no difference"
        diff = 1.0 - diff

        return diff

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

    if DEBUG_LEVEL >= 2:
        if match_score < 0.3:
            print(str(match_score))
            imshow(patch1, "patch1")
            imshow(patch2, "patch2")
            # imshow(match_score / 25565, "match_score")
            cv.waitKey(0)

    return match_score


if __name__ == "__main__":
    task2("Task2Dataset")
