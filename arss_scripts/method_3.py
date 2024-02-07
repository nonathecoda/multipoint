from pathlib import Path
from tqdm import tqdm
from icecream import ic
from kornia.feature import LoFTR
from PIL import Image
import numpy as np
import kornia as K
import kornia.feature as KF
from kornia_moons.viz import draw_LAF_matches
import cv2
import torch
import matplotlib.pyplot as plt
import os


def distance(p1, p2):
    """Calculate the Euclidean distance between two points."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def furthest_points(points, n=4):
    """Select N points that are the furthest from each other."""
    if len(points) <= n:
        return points

    # Start with the point with the maximum x-coordinate
    selected_points = [max(points, key=lambda point: point[0])]
    
    # Iteratively add the furthest point from the current selection
    while len(selected_points) < n:
        furthest_point = max(points, key=lambda point: min([distance(point, selected) for selected in selected_points]))
        selected_points.append(furthest_point)

    return selected_points

class ImageAligner():
    '''
    Class to iterate over a image pairs and align them using mutual information.
    '''

    def __init__(self, input_dir, output_dir):
        '''
        Initializer
        
            Parameters
        ----------
        input_dir : string
            Directory containing the preprocessed image pairs
        output_dir : string
            Directory where the processed images will be stored
        
        '''

        self.__input_dir = input_dir
        self.__output_dir = output_dir

        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        
        self.__optical_filenames = [f for f in os.listdir(input_dir)
                                    if os.path.isfile(os.path.join(input_dir, f)) and 'optical' in f]
        
        print('Number of pairs: ' + str(len(self.__optical_filenames)))

        self.__counter = dict()
        self.__counter['total'] = 0


    def run(self):

        out_path_aligned = Path(self.__output_dir, 'aligned')

        if not os.path.isdir(str(out_path_aligned)):
            os.makedirs(str(out_path_aligned))
        
        with tqdm(total=len(self.__optical_filenames)) as pbar:
            for optical_name in self.__optical_filenames:
                # get the image paths
                index = optical_name.split('_')[0]
                optical_path = Path(self.__input_dir, optical_name)
                thermal_path = Path(self.__input_dir, index + '_thermal.png')

                # load the images
                img_optical = K.io.load_image(optical_path, K.io.ImageLoadType.RGB32)[None, ...]
                img_thermal = K.io.load_image(thermal_path, K.io.ImageLoadType.RGB32)[None, ...]

                img_optical = K.geometry.resize(img_optical, (240, 320), antialias=True)
                img_thermal = K.geometry.resize(img_thermal, (240, 320), antialias=True)

                matcher = KF.LoFTR(pretrained="outdoor")

                input_dict = {
                    "image0": K.color.rgb_to_grayscale(img_optical),  # LofTR works on grayscale images only
                    "image1": K.color.rgb_to_grayscale(img_thermal),
                }

                with torch.inference_mode():
                    correspondences = matcher(input_dict)

                mkpts_optical = correspondences["keypoints0"].cpu().numpy()
                mkpts_thermal = correspondences["keypoints1"].cpu().numpy()
                Fm, inliers = cv2.findFundamentalMat(mkpts_optical, mkpts_thermal, cv2.USAC_MAGSAC, 0.5, 0.999, 100000)
                inliers = inliers > 0

                # Select inlying points
                mkpts_optical_inlier = []
                mkpts_thermal_inlier = []
                for i in range(len(inliers)):
                    if inliers[i]:
                        mkpts_optical_inlier.append(mkpts_optical[i])
                        mkpts_thermal_inlier.append(mkpts_thermal[i])

                mkpts_optical = np.asarray(mkpts_optical_inlier)
                mkpts_thermal = np.asarray(mkpts_thermal_inlier)

                # Select the 4 points that are the furthest from each other
                selected_points = furthest_points(mkpts_optical)

                # find and store index of selected points
                chosen_values = []
                index_ = None
                for selected_point in selected_points:
                    for i, point in enumerate(mkpts_optical):
                        if np.array_equal(point, selected_point):
                            index_ = i
                            break
                    chosen_values.append(index_)

                # Extract the selected keypoints
                keypoints_optical = mkpts_optical[chosen_values]
                keypoints_thermal = mkpts_thermal[chosen_values]

                inliers = np.array([[True], [True], [True], [True]])

                # find homography
                H = cv2.findHomography(keypoints_optical, keypoints_thermal, cv2.RANSAC)[0]

                # warp image
                img_optical_np = img_optical.numpy()
                img_optical_np = img_optical_np[0, 0, :, :]

                img_thermal_np = img_thermal.numpy()
                img_thermal_np = img_thermal_np[0, 0, :, :]

                img_optical_warped = cv2.warpPerspective(img_optical_np, H, (img_optical_np.shape[1], img_optical_np.shape[0]))

                cv2.imwrite(str(Path(out_path_aligned, index + '_optical.png')),
                            img_optical_warped*255)
                cv2.imwrite(str(Path(out_path_aligned, index + '_thermal.png')),
                            img_thermal_np*255)
                
                # update the progressbar
                pbar.update(1)
                
        

def main():

    input_dir = 'tmp/processed/preprocessed'
    output_dir = 'tmp/processed/method_3'

    worker = ImageAligner(input_dir, output_dir)

    worker.run()

if __name__ == "__main__":
    main()