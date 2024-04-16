import os
from pathlib import Path
from typing import List, Tuple
import cv2 as cv
from dataclasses import dataclass
import numpy as np

from ransac import Point, Ransac

def task3(folderName: str) -> float:
    # TODO: remove
    path_dir = os.getcwd() + "/datasets/IconDataset/png/01-lighthouse.png"
    img = cv.imread(str(path_dir))
    run(img=img)
 
    return 0.0

def run(img):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    sift = cv.SIFT_create() # ignore "known member" remember

    keypoints, descriptors = sift.detectAndCompute(img_gray, None)
    #datasets/Task3Dataset/images/test_image_1.png
    # converts the SIFT detectAndCompute keypoints to Point (dataclass) form for ransac
    points = []
    for keypoint in keypoints:
        points.append(Point(int(keypoint.pt[0]), int(keypoint.pt[1])))

    # img_keypoints_original = np.copy(img)  
    # img_keypoints_original = cv.drawKeypoints(img, keypoints, img_keypoints_original, color=(0, 255, 0), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv.imshow(" ", img_keypoints_original)
    # cv.waitKey(0)  
    # cv.destroyAllWindows()

    
    ransac = Ransac(distance_threshold=10, sample_points_num=30)  

    try:
        best_points, best_line = ransac.run_ransac(points=points, iterations=100) 
        print(f"number keypoints removed: {len(keypoints) - len(best_points)} / {len(keypoints)}")
    except ValueError as e:
        print(e)


if __name__ == "__main__":
    task3("Task3Dataset")
