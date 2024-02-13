import cv2
import numpy as np
from icecream import ic

def detect_aruco_markers(image):
    # Load the predefined dictionary
    ARUCO_DICT = cv2.aruco.DICT_7X7_100
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)

    # Initialize the detector parameters using default values
    #parameters = cv2.aruco.DetectorParameters_create()
    parameters =  cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    # Detect the markers in the image
    markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(image)

    ic(markerCorners, markerIds, rejectedCandidates)

    return markerCorners, markerIds

def find_homography(corners1, corners2):
    # Assuming corners1 and corners2 are the corresponding corners of the same ArUco marker in both images
    # Compute the homography matrix
    H, status = cv2.findHomography(corners1, corners2)
    return H

def warp_image(image, H, output_size):
    # Warp one image to the perspective of another
    aligned_image = cv2.warpPerspective(image, H, output_size)
    return aligned_image

# Load images
image1 = cv2.imread('/Users/antonia/dev/UNITN/remote_sensing_systems/multipoint/tmp/optical_aligned/cam1/cam1_00160_e_0012_g_01_98372962_corr_rect.png')
image2 = cv2.imread('/Users/antonia/dev/UNITN/remote_sensing_systems/multipoint/tmp/optical_aligned/cam2/cam2_00160_e_0012_g_01_98372962_corr_rect.png')

# Detect ArUco markers
corners1, ids1 = detect_aruco_markers(image1)
corners2, ids2 = detect_aruco_markers(image2)

# Assuming we know corners1[0] and corners2[0] are correspondences
H = find_homography(corners1[0], corners2[0])

# Warp image
output_size = (image1.shape[1], image1.shape[0])
aligned_image = warp_image(image2, H, output_size)

# Display or save your aligned image
cv2.imshow('Aligned Image', aligned_image)
cv2.waitKey(0)
cv2.destroyAllWindows()