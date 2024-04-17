from dataclasses import dataclass
import os
from pathlib import Path
import cv2 as cv
import numpy as np
import numpy.typing as npt

import ransac

@dataclass
class Point:
    x: int
    y: int


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
    best_matches: list[TemplateImageKeypointMatch] = []
    # for each template keypoint
    for i, template_descriptor in enumerate(described_template_keypoints.descriptors):
        min_ssd = float("inf")
        best_image_point_index = None
        # find the best matching keypoint in the image
        for j, image_descriptor in enumerate(described_image_keypoints.descriptors):
            ssd = compare_descriptors_ssd(template_descriptor,image_descriptor)
            if ssd < min_ssd:
                min_ssd = ssd
                best_image_point_index = j


        # add the best matching (template_point,image_point) pair with the ssd      
        if best_image_point_index:
            template_point = Point(described_template_keypoints.keypoints[i].pt[0],
                                      described_template_keypoints.keypoints[i].pt[1])
            
            image_point = Point(described_image_keypoints.keypoints[best_image_point_index].pt[0],
                                   described_image_keypoints.keypoints[best_image_point_index].pt[1])
            
            best_matches.append(TemplateImageKeypointMatch(min_ssd,
                                                           template_point, 
                                                           image_point))
            
        # else:
        #     raise ValueError("error")

    if len(best_matches) < 4:
        raise ValueError("Need at least 4 matches")
    
    # sort the best matches based on the lowest ssd
    best_matches.sort(key=lambda x: x.match_ssd)
    
    return best_matches

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
        point = Point(int(point[0]), int(point[1]))
        transformed_points.append(point)

    return transformed_points

def run(image, template) -> bool:
    template_gray = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    # 1. SIFT
    sift = cv.SIFT_create() # ignore "known member" error
    template_keypoints, template_descriptors = sift.detectAndCompute(template_gray, None)
    image_keypoints, image_descriptors = sift.detectAndCompute(image_gray, None)

    described_template_keypoints = DescribedKeypoints(keypoints=template_keypoints, descriptors=template_descriptors)
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
    best_matches = descriptor_point_match(described_template_keypoints, described_image_keypoints)

    for best_match in best_matches:
        cv.drawMarker(image, (int(best_match.image_point.x), int(best_match.image_point.y)), (0, 255, 0), markerType=cv.MARKER_CROSS, markerSize=10, thickness=1)

    cv.imshow("matches", image)
    cv.waitKey(0)

    # 3. Threshold to identify matches
    if len(best_matches) < 10:
        return False

    # 4. RANSAC to get robust homography
    rsc = ransac.Ransac(distance_threshold=2)
    homography, inliers, outliers = rsc.run_ransac(best_matches, iterations=2000)

    # 5. Apply homography to get bounding box for labelling

    # # get corners in template image
    template_corners = np.array([
        [0, 0],  # top left
        [template.shape[1], 0],  # top right
        [template.shape[1], template.shape[0]],  # bottom right
        [0, template.shape[0]]  # bottom left
    ])

    bbox = apply_homography_transform(homography, template_corners)
    # print(bbox)

    # # transform the corners in the template image using the homography
    # homography_transformed_points = apply_homography_transform(H=H, points=template_corners)

    # print(homography_transformed_points)
    
    # draw_points_on_image(image, inliers, (0, 255, 0))
    # draw_points_on_image(image, outliers, (0, 0, 255))
    draw_bbox(image, bbox)

    return True
    # draw_points_on_image(image, bbox)

    # draw_matches_on_image(image=image, 
    #                       image_keypoints=image_keypoints,
    #                       matches=bbox)
    
    # draw_matches_on_template(template=template, 
    #                       template_keypoints=template_keypoints,
    #                       matches=best_4_matches)

def draw_points_on_image(image, points, color=(0, 255, 0)):
    for point in points:
        cv.drawMarker(image, (point.x, point.y), color, markerType=cv.MARKER_CROSS, markerSize=10, thickness=1)

    cv.imshow("image", image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def draw_bbox(image, bbox):
    cv.rectangle(image, (bbox[0].x, bbox[0].y), (bbox[2].x, bbox[2].y), (255, 0, 0), thickness=2)
    cv.imshow("image", image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def task3(folderName: str):
    this_file = Path(__file__)
    datasets_folder = this_file.parent.parent.parent / "datasets"
    dataset_folder = datasets_folder / folderName

    images_path = dataset_folder / "images"
    annotations_path = dataset_folder / "annotations"
    icon_dataset_path = datasets_folder / "IconDataset" / "png"


    template_path = icon_dataset_path / "01-lighthouse.png"
    image_path = images_path / "test_image_2.png"

    # for file in tqdm(natsorted(os.listdir(images_path)), desc="test image"):
    #     # Load test image
    #     image = cv.imread(str(images_path / file))

    #     icons_in_image = {}
    #     csv_file = (annotations_path / file).with_suffix(".csv")
    #     annotations = pd.read_csv(csv_file)

    #     for label, top, left, bottom, right in annotations.values:
    #         icons_in_image[label] = Icon(top, left, bottom, right, label)

    #     print()
    #     print(f"Test image: {file}")
    #     print(f"Found matches:")
    #     for match in matches:
    #         correct_match = "YES:" if match.label in icons_in_image else "NO: "
    #         print(f"\t{correct_match} {match}")

    #     correct_icons_string = "\n\t".join([str(icon) for icon in icons_in_image.values()])
    #     print(f"Icons in image:\n\t{correct_icons_string}")

    #     labels_in_image = [icon.label for icon in icons_in_image.values()]
    #     correct_matches = [match.label for match in matches if match.label in labels_in_image]
    #     accuracy = len(correct_matches) / len(labels_in_image)
    #     print(f"Accuracy: {accuracy * 100}%")

    template = cv.imread(str(template_path))
    image = cv.imread(str(image_path))
    
    # cv.imshow("template", template)
    # cv.imshow("src_image", image)

    run(image, template)

    # sift = SIFT.SIFT(image)

# def task3_run(folderName:str):
#     path_dir_template = os.getcwd() + "/datasets/IconDataset/png/01-lighthouse.png"
#     path_dir_image = os.getcwd() + "/datasets/Task3Dataset/images/test_image_2.png"

#     template = cv.imread(str(path_dir_template))
#     image = cv.imread(str(path_dir_image))

#     run(image, template)

if __name__ == "__main__":
    task3("Task3Dataset")