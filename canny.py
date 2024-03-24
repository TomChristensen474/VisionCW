import cv2 as cv
import numpy as np







# from skimage.util import invert


# https://scikit-image.org/docs/stable/auto_examples/edges/plot_skeleton.html#sphx-glr-download-auto-examples-edges-plot-skeleton-py
def Canny(image):
    # grayscale image
    grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # gaussian = cv.GaussianBlur(grayscale, (9, 9), 0)
    # gaussian = image

    # cv.imshow("Gaussian", gaussian)
    # sobel = cv.Sobel(grayscale, cv.CV_8U, 1, 1, ksize=5)

    # # edges = cv.Canny(grayscale, 100, 200, None, 5)
    # cv.imshow("Sobel", sobel)

    # thresholded, weak, strong = threshold(sobel)

    # # np.uint8(thresholded)

    # # cv.imshow("Thresholded", thresholded)

    # canny = hysteresis(thresholded, 10, int(strong))

    # dilated = dilate(canny.astype(np.uint8))
    # cv.imshow("Dilated", dilated)

    # thinned = skeletonize(sobel.astype(np.uint8))
    grayscale = grayscale > 130
    skeleton = skeletonize(grayscale.astype(np.uint8))
    skeleton = skeleton.astype(np.uint8)*255
    # cv.imshow("Canny", canny)
    cv.imshow("Thinned", skeleton)

    cv.waitKey(0)

    return skeleton


def threshold(img, lowThresholdRatio=0.05, highThresholdRatio=0.09):
    """
    Double threshold
    """

    highThreshold = img.max() * highThresholdRatio
    lowThreshold = highThreshold * lowThresholdRatio

    M, N = img.shape
    res = np.zeros((M, N), dtype=np.int32)

    weak = np.int32(25)
    strong = np.int32(255)

    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)

    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return (res, weak, strong)


def hysteresis(img, weak, strong=255):
    M, N = img.shape

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if img[i, j] == weak:
                if (
                    (img[i + 1, j - 1] == strong)
                    or (img[i + 1, j] == strong)
                    or (img[i + 1, j + 1] == strong)
                    or (img[i, j - 1] == strong)
                    or (img[i, j + 1] == strong)
                    or (img[i - 1, j - 1] == strong)
                    or (img[i - 1, j] == strong)
                    or (img[i - 1, j + 1] == strong)
                ):
                    img[i, j] = strong
                else:
                    img[i, j] = 0
    return img


def skeletonize(img):
    skel = np.zeros(img.shape, np.uint8)

    element = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    # element2 = cv.getStructuringElement(cv.MORPH_CROSS, (2, 2))
    done = False

    while not done:
        eroded = cv.erode(img, element)
        temp = cv.dilate(eroded, element)
        temp = cv.subtract(img, temp)
        skel = cv.bitwise_or(skel, temp)
        img = eroded.copy()

        if cv.countNonZero(img) == 0:
            done = True

    return skel


def dilate(img):
    return cv.dilate(img, cv.getStructuringElement(cv.MORPH_CROSS, (5, 5)))


def erosion(image):
    image = cv.dilate(image, cv.getStructuringElement(cv.MORPH_CROSS, (5, 5)))

    erosion_dst = cv.erode(image, cv.getStructuringElement(cv.MORPH_CROSS, (10, 10)))

    return erosion_dst
