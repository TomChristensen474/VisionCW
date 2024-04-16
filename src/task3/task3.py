import os
from pathlib import Path
import cv2 as cv
import numpy as np
from ransac import Point, Ransac


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


def descriptor_ssd(descriptor1, descriptor2):
    pass

def run(image, template):
    template_gray = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    # 1. SIFT
    sift = cv.SIFT_create() # ignore "known member" remember
    template_keypoints, template_descriptors = sift.detectAndCompute(template_gray, None)
    image_keypoints, image_descriptors = sift.detectAndCompute(image_gray, None)


    # converts the SIFT detectAndCompute keypoints to Point (dataclass) form for ransac
    templatePointDescriptorMap = {}
    template_points = []
    for keypoint, descriptor in zip(template_keypoints, template_descriptors):
        point = Point(int(keypoint.pt[0]), int(keypoint.pt[1]))
        template_points.append(point)
        templatePointDescriptorMap[point] = descriptor

    
    # 2. RANSAC
    ransac = Ransac(distance_threshold=10, sample_points_num=30)  
    try:
        # get filtered points and the ransac line
        filtered_points, best_line = ransac.run_ransac(points=points, iterations=100) 

        # get the corresponding descriptors for each filtered point
        filtered_descriptors = np.array([pointDescriptorMap[point] for point in filtered_points])

        print(f"Keypoints removed: {len(keypoints) - len(filtered_points)} / {len(keypoints)}")

        if len(keypoints) - len(filtered_points) < 4:
            raise ValueError("need at least 4 keypoints for homography")

    except ValueError as e:
        print(e)

    template_points = get_template_points(input_points=filtered_points)
    image_points = get_image_points()

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