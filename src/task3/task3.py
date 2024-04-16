from dataclasses import dataclass
import os
from pathlib import Path
import cv2 as cv
import numpy as np
from ransac import Point, Ransac

@dataclass
class TemplateImageKeypointMatch:
    match_ssd: float
    template_point_index: int 
    image_point_index: int

# for point algorithm from lectures
def four_point_algorithm(p, q):
    A = np.zeros((8, 9))
    for i in range(4):
        A[2*i, 0:3] = p[:, i]
        A[2*i, 6:9] = -q[0, i]*p[:, i]
        A[2*i+1, 3:6] = p[:, i]
        A[2*i+1, 6:9] = -q[1, i]*p[:, i]

    # Solve the homogeneous linear system using SVD
    U, D, Vt = np.linalg.svd(A)
    H = Vt[-1, :].reshape(3, 3)

    # Normalize the solution to ensure H[2, 2] = 1
    H = H / H[2, 2]
    
    return H

def task3_run(folderName:str):
    path_dir_template = os.getcwd() + "/datasets/IconDataset/png/01-lighthouse.png"
    path_dir_image = os.getcwd() + "/datasets/Task3Dataset/images/test_image_2.png"

    template = cv.imread(str(path_dir_template))
    image = cv.imread(str(path_dir_image))

    run(image, template)

"""
Takes in SIFT points after RANSAC and chooses 4 points (template)
"""
def get_template_points(input_points: list[Point]):
    output_points = []
    return output_points

"""
Matches template points to image points using descirptors 
"""
def match_points_using_descriptors():
    pass

"""
Takes in SIFT points after RANSAC and chooses 4 points (image)
"""
def get_image_points(input_points: list[Point]):
    output_points = []
    return output_points


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
def descriptor_point_match(template_descriptors, image_descriptors):
    best_matches = []
    # for each template keypoint
    for i, template_descriptor in enumerate(template_descriptors):
        min_ssd = float("inf")
        best_image_point_index = None
        # find the best matching keypoint in the image
        for j, image_descriptor in enumerate(image_descriptors):
            ssd = compare_descriptors_ssd(template_descriptor,image_descriptor)
            if ssd < min_ssd:
                min_ssd = ssd
                best_image_point_index = j

        # add the best matching (template_point,image_point) pair with the ssd      
        if best_image_point_index:  
            best_matches.append(TemplateImageKeypointMatch(match_ssd=min_ssd, 
                                                       template_point_index=i, 
                                                       image_point_index=best_image_point_index))
        else:
            raise ValueError("error")


    # sort the best matches based on the lowest ssd
    best_matches.sort(key=lambda x: x.match_ssd)
    return best_matches

def run(image, template):
    template_gray = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    # 1. SIFT
    sift = cv.SIFT_create() # ignore "known member" remember
    template_keypoints, template_descriptors = sift.detectAndCompute(template_gray, None)
    image_keypoints, image_descriptors = sift.detectAndCompute(image_gray, None)


    template_points = []
    for keypoint in template_keypoints:
        template_points.append(Point(int(keypoint.pt[0]), int(keypoint.pt[1])))

    image_points = []
    for keypoint in image_keypoints:
        image_points.append(Point(int(keypoint.pt[0]), int(keypoint.pt[1])))

    # ransac to filter keypoints
    ransac = Ransac(distance_threshold=10, sample_points_num=30)
    filtered_template_points, _ = ransac.run_ransac(template_points, iterations=100)
    filtered_image_points, _ = ransac.run_ransac(image_points, iterations=100)


    # get filtered descriptors based on filtered RANSAC points
    filtered_template_descriptors = []
    for point in filtered_template_points:
        index = template_points.index(point)
        filtered_template_descriptors.append(template_descriptors[index])

    filtered_image_descriptors = []
    for point in filtered_image_points:
        index = image_points.index(point)
        filtered_image_descriptors.append(image_descriptors[index])
    
    best_matches = descriptor_point_match(filtered_template_descriptors, filtered_image_descriptors)

    # get the best 4 matches
    # TODO: may need checks to see if the 4 are good enough
    best_4_matches = best_matches[:4]

    print(best_4_matches)
    #H = four_point_algorithm(template_points, image_points)



def task3(folderName: str):
    this_file = Path(__file__)
    datasets_folder = this_file.parent.parent.parent / "datasets"
    dataset_folder = datasets_folder / folderName

    images_path = dataset_folder / "images"
    annotations_path = dataset_folder / "annotations"
    icon_dataset_path = datasets_folder / "IconDataset" / "png"

    image_path = icon_dataset_path / "01-lighthouse.png"
    image = cv.imread(str(image_path))

    # sift = SIFT.SIFT(image)


if __name__ == "__main__":
    #task3("Task3Dataset")
    task3_run("Task3Dataset")