import argparse
import cv2
import os
from pathlib import Path
from tqdm import tqdm
from icecream import ic
from kornia.feature import LoFTR
from PIL import Image
import numpy as np
import kornia as K
import kornia.feature as KF
from kornia_moons.viz import draw_LAF_matches
import torch
import matplotlib.pyplot as plt

## step 1: preprocess images
## step 2: match images based on pairs.txt file
class Preprocessor():
    '''
    Class to iterate over a bagfile and find image pairs based on the timestamp.
    '''

    def __init__(self, input_dir, output_dir):
        ic("hi")
        self.__output_dir = output_dir
        self.__input_dir = input_dir


    def run(self):
        
        out_path_aligned = Path(self.__output_dir, 'preprocessed')

        if not os.path.isdir(str(out_path_aligned)):
            os.makedirs(str(out_path_aligned))
        
        #iterate over input directory
        img__names = os.listdir(str(Path(self.__input_dir, 'cam1')))
        cams = ['cam1', 'cam2', 'cam3', 'cam4']
        for cam in cams:
            if cam == 'cam1':
                continue
            with tqdm(total=len(img__names)) as pbar:
                for img_path in img__names:
                    try:
                        img_path_1 = img_path
                        img_path_1 = str(Path(self.__input_dir, 'cam1', img_path_1))
                        img1 = cv2.imread(img_path_1)
                        img_path_0 = img_path.replace('cam1', cam)
                        img_path_0 = str(Path(self.__input_dir, cam, img_path_0))
                        img0 = cv2.imread(img_path_0)

                        # apply loftr

                        # load the images
                        ic(img_path_0)
                        ic(img_path_1)
                        img_0 = K.io.load_image(img_path_0, K.io.ImageLoadType.RGB32)[None, ...]
                        img_1 = K.io.load_image(img_path_1, K.io.ImageLoadType.RGB32)[None, ...]

                        #img_1 = K.geometry.resize(img_1, (240, 320), antialias=True)
                        #img_2 = K.geometry.resize(img_2, (240, 320), antialias=True)

                        matcher = KF.LoFTR(pretrained="outdoor")

                        input_dict = {
                            "image0": K.color.rgb_to_grayscale(img_0),  # LofTR works on grayscale images only
                            "image1": K.color.rgb_to_grayscale(img_1),
                        }

                        with torch.inference_mode():
                            correspondences = matcher(input_dict)

                        mkpts_0 = correspondences["keypoints0"].cpu().numpy()
                        mkpts_1 = correspondences["keypoints1"].cpu().numpy()
                        Fm, inliers = cv2.findFundamentalMat(mkpts_0, mkpts_1, cv2.USAC_MAGSAC, 0.5, 0.999, 100000)
                        inliers = inliers > 0

                        '''
                        # Select inlying points
                        mkpts_0_inlier = []
                        mkpts_1_inlier = []
                        for i in range(len(inliers)):
                            if inliers[i]:
                                mkpts_0_inlier.append(mkpts_0[i])
                                mkpts_1_inlier.append(mkpts_1[i])

                        mkpts_0 = np.asarray(mkpts_0_inlier)
                        mkpts_1 = np.asarray(mkpts_1_inlier)

                        # Select the 4 points that are the furthest from each other
                        selected_points = self.furthest_points(mkpts_0)

                        # find and store index of selected points
                        chosen_values = []
                        index_ = None
                        for selected_point in selected_points:
                            for i, point in enumerate(mkpts_0):
                                if np.array_equal(point, selected_point):
                                    index_ = i
                                    break
                            chosen_values.append(index_)

                        # Extract the selected keypoints
                        keypoints_0 = mkpts_0[chosen_values]
                        keypoints_1 = mkpts_1[chosen_values]

                        inliers = np.array([[True], [True], [True], [True]])
                        
                        
                        draw_LAF_matches(
                            KF.laf_from_center_scale_ori(
                                torch.from_numpy(mkpts_0).view(1, -1, 2),
                                torch.ones(mkpts_0.shape[0]).view(1, -1, 1, 1),
                                torch.ones(mkpts_0.shape[0]).view(1, -1, 1),
                            ),
                            KF.laf_from_center_scale_ori(
                                torch.from_numpy(mkpts_0).view(1, -1, 2),
                                torch.ones(mkpts_1.shape[0]).view(1, -1, 1, 1),
                                torch.ones(mkpts_1.shape[0]).view(1, -1, 1),
                            ),
                            torch.arange(mkpts_0.shape[0]).view(-1, 1).repeat(1, 2),
                            K.tensor_to_image(img_0),
                            K.tensor_to_image(img_1),
                            inliers,
                            draw_dict={"inlier_color": (0.2, 1, 0.2), "tentative_color": None, "feature_color": (0.2, 0.5, 1), "vertical": False},
                        )
                        #plt.show()
                        '''

                        # find homography
                        H = cv2.findHomography(mkpts_0, mkpts_1, cv2.RANSAC)[0]

                        # warp image
                        img_0_np = img_0.numpy()
                        img_0_np = img_0_np[0, 0, :, :]

                        img_1_np = img_1.numpy()
                        img_1_np = img_1_np[0, 0, :, :]

                        img_0_warped = cv2.warpPerspective(img_0_np, H, (img_0_np.shape[1], img_0_np.shape[0]))
                        
                        img_path = img_path.replace('tiff', 'png')
                        outpath_0 = img_path.replace('cam1', cam)
                        outpath_0 = str(Path(self.__output_dir, cam, outpath_0))                        
                        outpath_1 = str(Path(self.__output_dir, 'cam1', img_path))
                        
                        cv2.imwrite(outpath_0, img_0_warped*255)
                        cv2.imwrite(outpath_1, img_1_np*255)
                    except:
                        ic("error")

                    pbar.update(1)
            
    def distance(self, p1, p2):
        """Calculate the Euclidean distance between two points."""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def furthest_points(self, points, n=4):
        """Select N points that are the furthest from each other."""
        if len(points) <= n:
            return points

        # Start with the point with the maximum x-coordinate
        selected_points = [max(points, key=lambda point: point[0])]
        
        # Iteratively add the furthest point from the current selection
        while len(selected_points) < n:
            furthest_point = max(points, key=lambda point: min([self.distance(point, selected) for selected in selected_points]))
            selected_points.append(furthest_point)

        return selected_points


        
def main():
    parser = argparse.ArgumentParser(description='Extract images from a rosbag and save them as pairs')
    parser.add_argument('-i', '--input-dir', default='/Users/antonia/dev/UNITN/remote_sensing_systems/data/ARSS_P3/', help='Input directory to hdf5 files')
    parser.add_argument('-o', '--output-dir', default='/Users/antonia/dev/UNITN/remote_sensing_systems/multipoint/tmp/optical_aligned', help='Output directory')

    args = parser.parse_args()

    worker = Preprocessor(args.input_dir, args.output_dir)

    worker.run()

if __name__ == "__main__":
    main()
