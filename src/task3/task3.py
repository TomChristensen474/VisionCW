from dataclasses import dataclass
from natsort import natsorted
from pathlib import Path
from tqdm import tqdm

import cv2 as cv
from joblib import Parallel, delayed
import numpy as np
import numpy.typing as npt
import os
import pandas as pd
import ransac
import re


@dataclass
class Point:
    x: int
    y: int


@dataclass
class Metrics:
    TP: int
    TN: int
    FP: int
    FN: int


@dataclass
class Icon:
    top: int
    left: int
    bottom: int
    right: int
    label: str


@dataclass
class Homography:
    matrix: npt.NDArray[np.float64]  # 3x3 matrix


@dataclass
class TemplateImageKeypointMatch:
    match_ssd: float
    template_point: Point
    image_point: Point


@dataclass
class DescribedKeypoints:
    keypoints: list[cv.KeyPoint]
    descriptors: list[np.ndarray]


def print(s=""):
    tqdm.write(str(s))


def compare_descriptors_ssd(descriptor1, descriptor2):
    ssd_total = 0
    for i in range(len(descriptor1)):
        ssd = (descriptor1[i] - descriptor2[i]) ** 2
        ssd_total += ssd
    return ssd_total


"""
this will match descriptors in template with descriptors in image and return 
best matching points which are sorted based on lowest difference metric (e.g. lowest SSD)
"""


def descriptor_point_match(described_template_keypoints, described_image_keypoints):
    ssd_threshold = 120000
    r_value = 0.8
    matches: list[TemplateImageKeypointMatch] = []
    # for each template keypoint
    for i, template_descriptor in enumerate(described_template_keypoints.descriptors):
        min_ssd = float("inf")
        second_to_min_ssd = float("inf")
        best_image_point_index = None
        # find the best matching keypoint in the image
        for j, image_descriptor in enumerate(described_image_keypoints.descriptors):
            ssd = compare_descriptors_ssd(template_descriptor, image_descriptor)
            if ssd < min_ssd:
                second_to_min_ssd = min_ssd
                min_ssd = ssd
                best_image_point_index = j

        # add the best matching (template_point,image_point) pair with the ssd
        if min_ssd < ssd_threshold:
            if min_ssd / second_to_min_ssd < r_value:
                template_point = Point(
                    described_template_keypoints.keypoints[i].pt[0],
                    described_template_keypoints.keypoints[i].pt[1],
                )

                image_point = Point(
                    described_image_keypoints.keypoints[best_image_point_index].pt[0],
                    described_image_keypoints.keypoints[best_image_point_index].pt[1],
                )

                matches.append(TemplateImageKeypointMatch(min_ssd, template_point, image_point))

        # else:
        #     raise ValueError("error")

    # if len(matches) < 4:
    #     raise ValueError("Need at least 4 matches")

    # sort the best matches based on the lowest ssd
    matches.sort(key=lambda x: x.match_ssd)

    # return matches

    unique_matches = []
    source_points = []
    destination_points = []
    for match in matches:
        if match.template_point not in source_points:
            if match.image_point not in destination_points:
                unique_matches.append(match)
                source_points.append(match.template_point)
                destination_points.append(match.image_point)

    return unique_matches


"""
points:
[[x1,y1],
[x2,y2],
[x3,y3],
[x4,y4]]
"""


def apply_homography_transform(H: Homography, points: npt.NDArray[np.uint8]) -> list[Point]:
    """
    points_homogeneous
    [[x1 x2 x3 x4],
    [y1,y2,y3,y4],
    [1,1,1,1]]
    """
    points_homogeneous = np.vstack([points.T, np.ones((1, points.shape[0]))])

    transformed_points = H.matrix @ points_homogeneous

    transformed_points /= transformed_points[2, :]

    """
    points_cartesian_form
    [[x1,y1],
    [x2,y2],
    [x3,y3],
    [x4,y4]]
    """
    points_cartesian_form = transformed_points[:2, :].T

    transformed_points = []

    for point in points_cartesian_form:
        # if np.isinf(point[0]) or np.isinf(point[1]):
        #     point = Point(0, 0) # set to 0,0 if point is infinity
        # else:
        point = Point(int(point[0]), int(point[1]))
        transformed_points.append(point)

    return transformed_points


def run(image, template) -> tuple[bool, int, list[int]]:
    template_gray = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # 1. SIFT
    sift = cv.SIFT_create(nOctaveLayers=3)  # ignore "known member" error
    template_keypoints, template_descriptors = sift.detectAndCompute(template_gray, None)
    image_keypoints, image_descriptors = sift.detectAndCompute(image_gray, None)

    described_template_keypoints = DescribedKeypoints(
        keypoints=template_keypoints, descriptors=template_descriptors
    )
    described_image_keypoints = DescribedKeypoints(keypoints=image_keypoints, descriptors=image_descriptors)

    # for i in range(len(template_keypoints)):
    #     cv.drawMarker(template, (int(template_keypoints[i].pt[0]), int(template_keypoints[i].pt[1])), (0, 255, 0), markerType=cv.MARKER_CROSS, markerSize=10, thickness=1)
    # cv.imshow("template", template)
    # cv.waitKey(0)

    # for i in range(len(image_keypoints)):
    #     cv.drawMarker(image, (int(image_keypoints[i].pt[0]), int(image_keypoints[i].pt[1])), (0, 255, 0), markerType=cv.MARKER_CROSS, markerSize=10, thickness=1)

    # cv.imshow("image", image)
    # cv.waitKey(0)
    # get the best matches (image and corresponding template keypoint matches)
    # 2. Match keypoints
    matches = descriptor_point_match(described_template_keypoints, described_image_keypoints)
    match_image = image.copy()
    for best_match in matches:
        cv.drawMarker(
            match_image,
            (int(best_match.image_point.x), int(best_match.image_point.y)),
            (0, 255, 0),
            markerType=cv.MARKER_CROSS,
            markerSize=10,
            thickness=1,
        )

    # print(str(len(matches)))
    # 3. Threshold to identify matches -> need minimum 4 for homography
    if len(matches) < 4:
        return False, len(matches), []

    cv.imshow("matches", match_image)
    # cv.waitKey(0)

    # 4. RANSAC to get robust homography
    rsc = ransac.Ransac(distance_threshold=15)
    homography, inliers, outliers, sampled_inliers = rsc.run_ransac(matches, iterations=2000)

    # print(str(len(inliers) + len(outliers)))

    # # 5. Threshold to identify matches within inliers -> a decent estimate for matching
    if len(inliers) < 10:
        return False, len(inliers), []

    # get corners in template image
    template_corners = np.array(
        [
            [0, 0],  # top left
            [template.shape[1], 0],  # top right
            [template.shape[1], template.shape[0]],  # bottom right
            [0, template.shape[0]],  # bottom left
        ]
    )

    # 5. Apply homography to get bounding box for labelling
    bbox = apply_homography_transform(homography, template_corners)
    # print(bbox)
    img_copy = image.copy()
    sampled_inliers_img = image.copy()

    draw_outliers_on_image(img_copy, "outliers/inliers", inliers, (0, 255, 0))
    draw_outliers_on_image(img_copy, "outliers/inliers", outliers, (0, 0, 255))

    draw_outliers_on_image(sampled_inliers_img, "sampled_inliers", sampled_inliers, (255, 0, 255))
    draw_points_on_image(image, bbox)

    formatted_bbox_axes = [
        min(bbox[0].x, bbox[1].x, bbox[2].x, bbox[3].x),  # top
        min(bbox[0].y, bbox[1].y, bbox[2].y, bbox[3].y),  # left
        max(bbox[0].x, bbox[1].x, bbox[2].x, bbox[3].x),  # bottom
        max(bbox[0].y, bbox[1].y, bbox[2].y, bbox[3].y),  # right
    ]

    draw_axis_bbox(image, formatted_bbox_axes, (255, 0, 0))
    # cv.waitKey(0)  # uncomment to show images
    cv.destroyAllWindows()

    return True, len(matches), formatted_bbox_axes


def draw_outliers_on_image(image, label, outliers, color=(0, 255, 0)):
    for outlier in outliers:
        cv.drawMarker(
            image,
            (int(outlier.image_point.x), int(outlier.image_point.y)),
            color,
            markerType=cv.MARKER_CROSS,
            markerSize=10,
            thickness=1,
        )

    cv.imshow(label, image)


def draw_points_on_image(image, points, color=(255, 0, 0)):
    for point in points:
        cv.drawMarker(
            image, (point.x, point.y), color, markerType=cv.MARKER_CROSS, markerSize=10, thickness=2
        )

    cv.imshow("image", image)


def draw_axis_bbox(image, bbox_axes: list[int], color=(255, 0, 0)):
    # top, left, bottom, right
    cv.rectangle(image, (bbox_axes[0], bbox_axes[1]), (bbox_axes[2], bbox_axes[3]), color, thickness=2)
    cv.imshow("image", image)


def calc_iou(icon: Icon, bbox: list[int]) -> float:
    match_top, match_left, match_bottom, match_right = bbox
    top, left, bottom, right = icon.top, icon.left, icon.bottom, icon.right

    intersection_top = max(top, match_top)
    intersection_left = max(left, match_left)
    intersection_bottom = min(bottom, match_bottom)
    intersection_right = min(right, match_right)

    intersection_area = max(0, intersection_bottom - intersection_top) * max(
        0, intersection_right - intersection_left
    )

    match_area = (match_bottom - match_top) * (match_right - match_left)
    icon_area = (bottom - top) * (right - left)

    union_area = match_area + icon_area - intersection_area

    return intersection_area / union_area


def task3(folderName: str):
    this_file = Path(__file__)
    datasets_folder = this_file.parent.parent.parent / "datasets"
    dataset_folder = datasets_folder / folderName

    images_path = dataset_folder / "images"
    annotations_path = dataset_folder / "annotations"
    icon_dataset_path = datasets_folder / "IconDataset" / "png"

    # # for testing purposes
    # template_path = icon_dataset_path / "48-hospital.png"
    # image_path = images_path / "test_image_1.png"
    # template = cv.imread(str(template_path))
    # image = cv.imread(str(image_path))
    # run(image, template)

    total_accuracy = 0

    for file in tqdm(natsorted(os.listdir(images_path)), desc="test image"):
        # Load test image
        image = cv.imread(str(images_path / file))

        # Load annotations data
        icons_in_image = {}
        csv_file = (annotations_path / file).with_suffix(".csv")
        annotations = pd.read_csv(csv_file)

        for label, top, left, bottom, right in annotations.values:
            icons_in_image[label] = Icon(top, left, bottom, right, label)

        print(f"Test image: {file}")

        @delayed
        def detect_match_with_icon(icon) -> tuple[str, int, bool, bool, float|None, Metrics]:
            TP, TN, FP, FN = 0, 0, 0, 0
            template = cv.imread(str(icon_dataset_path / icon))
            icon_name = re.split("(\d+)-(.+)\.png", icon)[2]
            clean_image_copy = image.copy()  # creating clean copy of image for displaying
            match, num_matches, bbox = run(clean_image_copy, template)

            if icon_name in icons_in_image.keys():  # icon is in image
                correct_match = True
                if not match:  # False negative - icon not found
                    FN += 1
                    return icon_name, num_matches, match, correct_match, None, Metrics(TP, TN, FP, FN)

                # True positive
                ground_truth_bbox = [
                    icons_in_image[icon_name].top,
                    icons_in_image[icon_name].left,
                    icons_in_image[icon_name].bottom,
                    icons_in_image[icon_name].right,
                ]
                draw_axis_bbox(clean_image_copy, ground_truth_bbox, (0, 255, 0))

                iou = calc_iou(icons_in_image[icon_name], bbox)

                if iou > 0.5:
                    TP += 1  # True positive - made the right match
                else:
                    FN += 1  # False positive - made the wrong match

            else:
                correct_match = False
                iou = None  # icon not in image
                if match:  # False positive - not in image
                    FP += 1
                else:  # True negative
                    TN += 1

            metrics = Metrics(TP, TN, FP, FN)

            return icon_name, num_matches, match, correct_match, iou, metrics

        # for icon in tqdm(natsorted(os.listdir(icon_dataset_path)), desc="icon"):
        results_generator = Parallel(n_jobs=-1, return_as="generator")(
            detect_match_with_icon(icon) for icon in natsorted(os.listdir(icon_dataset_path))
        )

        total_accuracy, total_tpr, total_fpr, total_fnr = 0, 0, 0, 0
        TP, TN, FP, FN = 0, 0, 0, 0

        # results_generator = tqdm(results_generator, total=len(os.listdir(icon_dataset_path)), desc="icon")
        for result in results_generator:
            if result is None:
                print("None returned")
                continue
            icon_name, num_matches, predicted_match, correct_match, iou, metrics = result
            if iou:
                print(
                    f"Icon: {icon_name}, Matches: {num_matches}, Predicted_match: {iou > 0.5}, Correct_match: {correct_match}, IOU: {iou}"
                )
            else:
                print(
                    f"Icon: {icon_name}, Matches: {num_matches}, Predicted_match: {predicted_match}, Correct_match: {correct_match}"
                )


                
            TP += metrics.TP
            TN += metrics.TN
            FP += metrics.FP
            FN += metrics.FN

        accuracy = (TP + TN) / (TP + TN + FP + FN) * 100
        tpr = TP / (TP + FN) * 100
        fpr = FP / (FP + TN) * 100
        fnr = FN / (TP + FN) * 100

        print(f"Accuracy: {accuracy}%, TPR: {round(tpr, 2)}%, FPR: {round(fpr, 2)}%, FNR: {round(fnr, 2)}%")

        total_accuracy += accuracy
        total_tpr += tpr
        total_fpr += fpr
        total_fnr = fnr

    average_accuracy = total_accuracy / len(os.listdir(images_path))
    average_tpr = total_tpr / len(os.listdir(images_path))
    average_fpr = total_fpr / len(os.listdir(images_path))
    print(f"Average Accuracy: {average_accuracy}%, Average TPR: {average_tpr}%, Average FPR: {average_fpr}%, FNR: {total_fnr}%")


if __name__ == "__main__":
    task3("Task3Dataset")
