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
    path_dir = os.getcwd() + "/datasets/IconDataset/png/01-lighthouse.png"
    img = cv.imread(str(path_dir))
    run(img=img)

def run(img):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # 1. SIFT
    sift = cv.SIFT_create() # ignore "known member" remember
    keypoints, descriptors = sift.detectAndCompute(img_gray, None)

    # converts the SIFT detectAndCompute keypoints to Point (dataclass) form for ransac
    points = []
    for keypoint in keypoints:
        points.append(Point(int(keypoint.pt[0]), int(keypoint.pt[1])))

    # img_keypoints_original = np.copy(img)  
    # img_keypoints_original = cv.drawKeypoints(img, keypoints, img_keypoints_original, color=(0, 255, 0), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv.imshow(" ", img_keypoints_original)
    # cv.waitKey(0)  
    # cv.destroyAllWindows()
    
    # 2. RANSAC
    ransac = Ransac(distance_threshold=10, sample_points_num=30)  
    try:
        best_points, best_line = ransac.run_ransac(points=points, iterations=100) 
        print(f"Keypoints removed: {len(keypoints) - len(best_points)} / {len(keypoints)}")

        if len(keypoints) - len(best_points) < 4:
            raise ValueError("need at least 4 keypoints for homography")

    except ValueError as e:
        print(e)




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