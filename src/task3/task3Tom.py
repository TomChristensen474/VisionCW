from dataclasses import dataclass
import os
from pathlib import Path
import cv2 as cv
import numpy as np
import ransacTom

@dataclass
class Point:
    x: int
    y: int

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
    best_matches = []
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
                                   described_image_keypoints.keypoints[best_image_point_index].pt[0])
            
            best_matches.append(TemplateImageKeypointMatch(min_ssd,
                                                           template_point, 
                                                           image_point))
            
        else:
            raise ValueError("error")


    # sort the best matches based on the lowest ssd
    best_matches.sort(key=lambda x: x.match_ssd)

    if len(best_matches) < 4:
        raise ValueError("Need at least 4 matches")
    
    return best_matches

def run(image, template):
    template_gray = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    # 1. SIFT
    sift = cv.SIFT_create() # ignore "known member" error
    template_keypoints, template_descriptors = sift.detectAndCompute(template_gray, None)
    image_keypoints, image_descriptors = sift.detectAndCompute(image_gray, None)

    described_template_keypoints = DescribedKeypoints(keypoints=template_keypoints, descriptors=template_descriptors)
    described_image_keypoints = DescribedKeypoints(keypoints=image_keypoints, descriptors=image_descriptors)

    # get the best matches (image and corresponding template keypoint matches)
    best_matches = descriptor_point_match(described_template_keypoints, described_image_keypoints)
    
    print(best_matches)
    print("")
    
    # 2. RANSAC
    ransac = ransacTom.Ransac(distance_threshold=20)
    ransac.run_ransac(best_matches, iterations=100)


    # # get corners in template image
    # template_corners = np.array([
    #     [0, 0],  # top left
    #     [template.shape[1], 0],  # top right
    #     [template.shape[1], template.shape[0]],  # bottom right
    #     [0, template.shape[0]]  # bottom left
    # ])

    # # transform the corners in the template image using the homography
    # homography_transformed_points = apply_homography_transform(H=H, points=template_corners)

    # print(homography_transformed_points)

    # draw_matches_on_image(image=image, 
    #                       image_keypoints=image_keypoints,
    #                       matches=best_4_matches)
    
    # draw_matches_on_template(template=template, 
    #                       template_keypoints=template_keypoints,
    #                       matches=best_4_matches)

"""
draws the keypoints on the template keypoints which matched to image keypoints
"""
def draw_matches_on_template(template, template_keypoints, matches):
    for match in matches:
        template_point = template_keypoints[match.template_point_index]
        cv.drawMarker(template, (int(template_point.pt[0]), int(template_point.pt[1])), (0, 255, 0), markerType=cv.MARKER_CROSS, markerSize=10, thickness=1)

    cv.imshow(" ", template)
    cv.waitKey(0)
    cv.destroyAllWindows()

"""
draws the keypoints on the image which matched to keypoints on templates
"""
def draw_matches_on_image(image, image_keypoints, matches):
    print(len(matches))
    for match in matches:
        image_point = image_keypoints[match.image_point_index]
        print(int(image_point.pt[0]))
        print(int(image_point.pt[1]))
        print("")
        cv.drawMarker(image, (int(image_point.pt[0]), int(image_point.pt[1])), (0, 255, 0), markerType=cv.MARKER_CROSS, markerSize=10, thickness=1)

    cv.imshow(" ", image)
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
    template = cv.imread(str(template_path))
    image = cv.imread(str(image_path))

    # sift = SIFT.SIFT(image)

# def task3_run(folderName:str):
#     path_dir_template = os.getcwd() + "/datasets/IconDataset/png/01-lighthouse.png"
#     path_dir_image = os.getcwd() + "/datasets/Task3Dataset/images/test_image_2.png"

#     template = cv.imread(str(path_dir_template))
#     image = cv.imread(str(path_dir_image))

#     run(image, template)

if __name__ == "__main__":
    task3("Task3Dataset")