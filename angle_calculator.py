import argparse
import cv2
import numpy as np
import pandas as pd
import os
import math
from matplotlib import pyplot as plt
from canny import Canny

def get_intersection_point(line1, line2):
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]

    # Check for parallel lines
    if np.abs(theta1 - theta2) < 1e-1:
        return None

    a1 = np.cos(theta1)
    b1 = np.sin(theta1)
    c1 = rho1
    a2 = np.cos(theta2)
    b2 = np.sin(theta2)
    c2 = rho2

    # Solve for intersection point
    denominator = a1 * b2 - a2 * b1
    if np.abs(denominator) < 1e-6:
        return None
    x = (c1 * b2 - c2 * b1) / denominator
    y = (a1 * c2 - a2 * c1) / denominator
    return (round(x), round(y))


def get_mean_intersection(img, lines):
    """
    This function takes a list of lines in (rho, theta) format and returns the mean intersection point.

    Args:
        lines: A list of tuples containing (rho, theta) for each line segment.

    Returns:
        An array containing (x, y) coordinates of the mean intersection point
    """
    intersections = []
    for i in range(len(lines) - 1):
        for j in range(i + 1, len(lines)):
            intersection = get_intersection_point(lines[i], lines[j])
            if intersection:
                intersections.append(intersection)

    # if not intersections:
    #     return None
    for x, y in intersections:
        cv2.circle(img, (x, y), 3, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imshow("intersections", img)

    # Calculate mean of intersection points
    x_sum, y_sum = 0, 0
    for x, y in intersections:
        x_sum += x
        y_sum += y
    mean_x = int(round(x_sum / len(intersections)))
    mean_y = int(round(y_sum / len(intersections)))
    #   return intersections
    return [mean_x, mean_y]


def get_non_black_pixels(img):
    # Threshold for black (adjust if needed)
    black_thresh = 10

    # Find non-zero pixels (non-black)
    non_black_pixels = np.where(img > black_thresh)

    # Transpose to get individual coordinates in a list of tuples
    return list(zip(*non_black_pixels))



def get_pixel_vectors(image, point):
    vectors = []
    white_pixels = get_non_black_pixels(image)

    # x and y is flipped because we transpose in get_non_black_pixels
    for white_pixel in white_pixels:
        cv2.circle(image, (white_pixel[1], white_pixel[0]), 1, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.imshow("name", image)

    for white_pixel in white_pixels:
        x = white_pixel[1]
        y = white_pixel[0]
        dx = x - point[0]
        dy = y - point[1]
        # Avoid division by zero for point itself
        if dx == 0 and dy == 0:
            vectors.append(np.array([0, 0]))
            # pass
        else:
            # Normalize the vector
            magnitude = np.sqrt(dx**2 + dy**2)
            # normalized_vector = np.array([dx / magnitude, dy / magnitude]) # TODO PUT THIS BACK IN
            vectors.append(np.array([dx, dy]))
    return vectors


def kmeans_cluster_directions(vectors, k):
    """
    As seen in docs: https://docs.opencv.org/3.4/d1/d5c/tutorial_py_kmeans_opencv.html
    """
    # Convert vectors to float32 for KMeans
    data = np.float32(vectors).reshape(-1, 1, 2)

    # Define termination criteria as seen in
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1)

    # Perform KMeans clustering
    _, _, centers = cv2.kmeans(data, k, data, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    return centers.reshape(k, 2)


def get_cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    return dot_product / (magnitude_v1 * magnitude_v2)


def detect_acute(img, intersection, debug_mode=False):
    vectors = get_pixel_vectors(img, intersection)
    for vector in vectors:
        cv2.arrowedLine(img, intersection, (intersection[0] + vector[0], intersection[1] + vector[1]), (0, 255, 0), 1)

    if debug_mode:
        cv2.imshow("Vectors", img)
    centers = kmeans_cluster_directions(vectors, 2)

    for vector in centers:
        cv2.arrowedLine(img, intersection, (intersection[0] + int(vector[0]), intersection[1] + int(vector[1])), (0, 0, 255), 1)
    
    if debug_mode:
        cv2.imshow("Vectors", img)
        
    if get_cosine_similarity(centers[0], centers[1]) >= 0:
        return True

    return False


def get_thetas_from_hough_lines(lines, img, debug_mode=False):
    thetas = []

    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        thetas.append(math.degrees(theta))
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))

        cv2.line(img, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)

    return thetas


def filter_similar_thetas(thetas):
    for theta in thetas:
        removed_thetas = []
        thetas_to_average = []
        removing = False

        for t in thetas:
            if abs(theta - t) < 5:
                removed_thetas.append(t)
                thetas_to_average.append(t)
                # thetas.remove(t)
                removing = True

            elif abs(theta - (t - 180)) < 5:
                removed_thetas.append(t)
                thetas_to_average.append(t - 180)
                removing = True

        if removing:
            mean_thetas = sum(thetas_to_average) / len(thetas_to_average)
            if mean_thetas < 0:
                mean_thetas += 180
            thetas.append(mean_thetas)
            for t in removed_thetas:
                thetas.remove(t)
    return thetas


def calculate_angle(thetas, acute):
    if acute:
        if thetas[1] - thetas[0] > 90:
            angle = 180 - (thetas[1] - thetas[0])
        else:
            angle = thetas[1] - thetas[0]
    else:
        if thetas[1] - thetas[0] > 90:
            angle = thetas[1] - thetas[0]
        else:
            angle = 180 - (thetas[1] - thetas[0])

    return abs(angle)


def get_angles_in_image(img, debug_mode=False):

    # edges = cv2.Canny(gaussian, 100, 200, None, 5)
    edges = Canny(img)

    if debug_mode:
        cv2.imshow("Canny", edges)
        cv2.waitKey(0)

    cdst = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 75)

    assert lines is not None, "No lines detected"

    mean_intersection = get_mean_intersection(cdst, lines)

    acute = detect_acute(cdst, mean_intersection, debug_mode)

    thetas = get_thetas_from_hough_lines(lines, cdst, debug_mode)

    cv2.circle(cdst, mean_intersection, 3, (0, 255, 0), 1, cv2.LINE_AA)

    if debug_mode:
        cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
        # cv2.imshow("Original", img)
        cv2.waitKey(0)

    cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
    # cv2.imshow("Original", img)
    cv2.waitKey(0)

    thetas.sort()
    thetas = filter_similar_thetas(thetas)

    assert len(thetas) == 2, "Error: More or less than 2 lines calculated"

    return calculate_angle(thetas, acute)
