
import cv2
import numpy as np

from thinner import Thinner

class EdgeDetector:
    def __init__(self):
        pass


    def dectect_edges(self, image_binary):
        raise NotImplementedError


if __name__ == "__main__":
    image = cv2.imread('./Task1Dataset/image4.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 120, 255, cv2.THRESH_BINARY)

    thinner = Thinner()
    thinned_img = thinner.thin_image(binary_image)
    edged_image = EdgeDetector().dectect_edges(image_binary=thinned_img)

    # cv2.imwrite("binary_image.png", binary_image)
    # cv2.imwrite("thinned_image.png", thinned_img)


    