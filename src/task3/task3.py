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
    """
    Estimate a 2D transformation using the four point algorithm.
    Args:
    p (ndarray): 3x4 array of homogeneous coordinates of points in the first image.
    q (ndarray): 3x4 array of homogeneous coordinates of corresponding points in the second image.
    Returns: H (ndarray): 3x3 transformation matrix that maps points in the first image to their corresponding points in the second image.
    """
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

    if len(best_matches) < 4:
        raise ValueError("Need at least 4 matches")
    
    return best_matches

def get_keypoints_and_descriptors_from_ransac(template_keypoints, image_keypoints, template_descriptors, image_descriptors):
    template_points = []
    for keypoint in template_keypoints:
        template_points.append(Point(int(keypoint.pt[0]), int(keypoint.pt[1])))

    image_points = []
    for keypoint in image_keypoints:
        image_points.append(Point(int(keypoint.pt[0]), int(keypoint.pt[1])))

    # ransac to filter keypoints
    ransac = Ransac(distance_threshold=20, sample_points_num=30)
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

    return template_points, image_points, filtered_template_descriptors, filtered_image_descriptors

"""
points:
[[x1,y1],
[x2,y2],
[x3,y3],
[x4,y4]]
"""
def apply_homography_transform(H, points):
    """
    points_homogeneous
    [[x1 x2 x3 x4],
    [y1,y2,y3,y4],
    [1,1,1,1]]
    """
    points_homogeneous = np.vstack([points.T, np.ones((1, points.shape[0]))])

    transformed_points = H @ points_homogeneous
    
    transformed_points /= transformed_points[2, :] 

    """
    points_cartesian_form
    [[x1,y1],
    [x2,y2],
    [x3,y3],
    [x4,y4]]
    """
    points_cartesian_form = transformed_points[:2, :].T

    return points_cartesian_form

def p_q_formatter(best_matches, template_keypoints, image_keypoints):
    if len(best_matches) != 4:
        raise ValueError("Need exactly 4 matches")

    # prepare points into homogenous coordinates for the homography 
    p = np.zeros((3, 4))
    q = np.zeros((3, 4))
    for i, match in enumerate(best_matches):  
        template_point = template_keypoints[match.template_point_index]
        image_point = image_keypoints[match.image_point_index]

        p[:, i] = np.array([template_point.pt[0], template_point.pt[1], 1])
        q[:, i] = np.array([image_point.pt[0], image_point.pt[1], 1])
    
    return p,q

def run(image, template):
    template_gray = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    # 1. SIFT
    sift = cv.SIFT_create() # ignore "known member" error
    template_keypoints, template_descriptors = sift.detectAndCompute(template_gray, None)
    image_keypoints, image_descriptors = sift.detectAndCompute(image_gray, None)

    # get the best 4 matches (image and corresponding template keypoint matches)
    best_matches = descriptor_point_match(template_descriptors, image_descriptors)
    
    print(best_matches)
    print("")
    #best_4_matches = best_matches[:4]

    best_4_matches = [best_matches[0],best_matches[3],best_matches[6],best_matches[10]]

    # format the template and image keypoints into the p,q form
    # to be used for the four point algorithm
    p,q = p_q_formatter(best_matches=best_4_matches, 
                        template_keypoints=template_keypoints, 
                        image_keypoints=image_keypoints)
    
    # homography matrix
    H = four_point_algorithm(p, q)

    # get corners in template image
    template_corners = np.array([
        [0, 0],  # top left
        [template.shape[1], 0],  # top right
        [template.shape[1], template.shape[0]],  # bottom right
        [0, template.shape[0]]  # bottom left
    ])

    # transform the corners in the template image using the homography
    homography_transformed_points = apply_homography_transform(H=H, points=template_corners)

    # convert to integers
    homography_transformed_points_ints = np.asarray(homography_transformed_points).astype(int)
    print(homography_transformed_points_ints)

    # draw_matches_on_image(image=image, 
    #                       image_keypoints=image_keypoints,
    #                       matches=best_4_matches)
    
    draw_int_coordinates(img=image,coords=homography_transformed_points_ints)

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

def draw_int_coordinates(img, coords):
    for coord in coords:
        x,y = coord[0], coord[1]
        # print(int(image_point.pt[0]))
        # print(int(image_point.pt[1]))
        # print("")
        cv.drawMarker(img, (int(x), int(y)), (0, 255, 0), markerType=cv.MARKER_CROSS, markerSize=10, thickness=1)

    cv.imshow(" ", img)
    cv.waitKey(0)
    cv.destroyAllWindows()

"""
draws the keypoints on the image which matched to keypoints on templates
"""
def draw_matches_on_image(image, image_keypoints, matches):
    print(len(matches))
    for match in matches:
        image_point = image_keypoints[match.image_point_index]
        # print(int(image_point.pt[0]))
        # print(int(image_point.pt[1]))
        # print("")
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

    image_path = icon_dataset_path / "01-lighthouse.png"
    image = cv.imread(str(image_path))

    # sift = SIFT.SIFT(image)


if __name__ == "__main__":
    #task3("Task3Dataset")
    task3_run("Task3Dataset")