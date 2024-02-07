from matplotlib import pyplot as plt
import argparse
import collections
import cv2
import logging
import math
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import yaml
from icecream import ic
import h5py

from helper_functions.align import align_images
from helper_functions.disp import *
from helper_functions.utils import *

# adapted from https://github.com/ethz-asl/multipoint
# finds homography between two images

class ImageAligner():
    '''
    Class to iterate over a image pairs and align them using mutual information.
    '''

    def __init__(self, input_dir, output_dir, config_file):
        '''
        Initializer
        
            Parameters
        ----------
        input_dir : string
            Directory containing the preprocessed image pairs
        output_dir : string
            Directory where the processed images will be stored
        bagfile : string
            Name of the bagfile to process
        config_file : string
            Yaml file specifying the alignment process
        '''

        self.__input_dir = input_dir
        self.__output_dir = output_dir

        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        with open(config_file, 'rt') as fh:
            self.__params = yaml.safe_load(fh)
        
        self.__optical_filenames = [f for f in os.listdir(input_dir)
                                    if os.path.isfile(os.path.join(input_dir, f)) and 'optical' in f]
        
        print('Number of pairs: ' + str(len(self.__optical_filenames)))

        self.__counter = dict()
        self.__counter['total'] = 0


    def run(self):

        #file_training = "/Users/antonia/dev/UNITN/remote_sensing_systems/arss_aerial_feature_matching/groundtruth_gui/data/training_del.hdf5"


        # create directories if required
        if self.__params['save_aligned_images']:
            out_path_aligned = Path(self.__output_dir, 'aligned')
    
            if not os.path.isdir(str(out_path_aligned)):
                os.makedirs(str(out_path_aligned))
        
        with tqdm(total=len(self.__optical_filenames)) as pbar:
            for optical_name in self.__optical_filenames:
                # get the image paths
                ic(optical_name)
                index = optical_name.split('_')[0]
                optical_path = Path(self.__input_dir, optical_name)
                thermal_path = Path(self.__input_dir, index + '_thermal.png')
                thermal_raw_path = Path(self.__input_dir, index + '_thermal_raw.png')

                # load the images
                optical = cv2.imread(str(optical_path), 0)
                thermal = cv2.imread(str(thermal_path), 0)

                #resize optical to thermal
                optical = cv2.resize(optical, (thermal.shape[1], thermal.shape[0]))

                if self.__params['LaPlacian']:
                    #apply gaussian Blur to remove the Gaussian Noise from the image
                    optical_blurr = cv2.GaussianBlur(optical,(5,5),0)
                    edges_optical = cv2.Laplacian(optical_blurr, -1, ksize=5, scale=1,delta=0,borderType=cv2.BORDER_DEFAULT)

                    thermal_blurr = cv2.GaussianBlur(thermal,(7,7),0)
                    edges_thermal = cv2.Laplacian(thermal_blurr, -1, ksize=5, scale=1,delta=0,borderType=cv2.BORDER_DEFAULT)
                
                elif self.__params['CannyEdge']:
                    
                    edges_optical = cv2.Canny(optical, 20, 70)
                    thermal_copy = (thermal).astype(np.uint8)
                    edges_thermal = cv2.Canny(thermal_copy, 70, 200)
                
                #plot edges
                plt.imshow((edges_thermal), cmap='gray')
                plt.show()
                exit()


                t_init = np.array([[1., 0, 0],
                                [0, 1.001, 0],
                                [0, 0, 0.001]])
                
                (success, solver_fail, best_warped, transformation,
                        valid_warped) = self.align_images_mutual_information(edges_optical, edges_thermal, t_init)
                
                ic(success)
                ic(transformation)
                

                #transformation = np.eye(3)
                best_warped = cv2.warpPerspective(optical, transformation, (optical.shape[1], optical.shape[0]))

                '''
                #visualize in plt
                fig, ax = plt.subplots(1, 2, figsize=(12, 6))
                ax[0].imshow(thermal, cmap='gray')
                ax[0].set_title('optical')
                ax[1].imshow(best_warped, cmap='gray')
                ax[1].set_title('best_warped')
                plt.show()
                '''

                if success:
                    if self.__params['save_aligned_images']:
                        cv2.imwrite(str(Path(out_path_aligned, index + '_optical.png')),
                                    best_warped)
                        cv2.imwrite(str(Path(out_path_aligned, index + '_thermal.png')),
                                    thermal)
                        ic("wrote to file " + str(Path(out_path_aligned, index + '_optical.png')))
                else:
                    if solver_fail:
                        logging.warning('%s', optical_name)

                # update the progressbar
                pbar.update(1)

    def align_images_mutual_information(self, optical, thermal, t_init):
        transformation = np.copy(t_init)
        success = True

        # refine the image with the full scale original image
        is_t_init = transformation == t_init
        (success, solver_fail, best_warped, transformation, valid_warped,
         self.__counter) = align_images(optical,
                                        thermal.astype(np.float32) / 65535.0,
                                        transformation,
                                        self.__params,
                                        self.__counter,
                                        self.__params['verbose'],
                                        self.__params['show_results'],
                                        filter_images=False)

        return success, solver_fail, best_warped, transformation, valid_warped

def main():
    parser = argparse.ArgumentParser(description='Extract and align the images from a rosbag')
    parser.add_argument('-y', '--yaml-config', default='arss_scripts/image_alignment/configs/config_align_images.yaml', help='Yaml file containing the configs')
    parser.add_argument('-i', '--input-dir', default='tmp/processed/preprocessed', help='Input directory')
    parser.add_argument('-o', '--output-dir', default='tmp/processed/method_4', help='Output directory')

    args = parser.parse_args()

    worker = ImageAligner(args.input_dir, args.output_dir, args.yaml_config)

    worker.run()

if __name__ == "__main__":
    main()