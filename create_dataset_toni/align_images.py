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

        file_training = "/Users/antonia/dev/UNITN/remote_sensing_systems/arss_aerial_feature_matching/groundtruth_gui/data/training_del.hdf5"

        with h5py.File(file_training, 'r') as f:
            keyzero = list(f.keys())[5]
            img_optical = np.array(f[keyzero]['optical'])
            img_thermal = np.array(f[keyzero]['thermal'])


            thermal_raw = None

            t_init = np.array([[1., 0, 0],
                               [0, 1.001, 0]])
            

            (success, solver_fail, best_warped, transformation,
                       valid_warped) = self.align_images_mutual_information(img_optical, img_thermal, thermal_raw, t_init)

            f, axarr = plt.subplots(2,1, figsize=(15, 5))
            axarr[0].imshow(img_optical, cmap='gray')
            axarr[0].imshow(img_thermal, cmap='gray', alpha=0.7)
            axarr[1].imshow(best_warped, cmap='gray')
            axarr[1].imshow(img_thermal, cmap='gray', alpha=0.7)
            plt.show()

        exit()
        
        # create directories if required
        if self.__params['save_aligned_images']:
            out_path_aligned_best = Path(self.__output_dir, 'aligned', 'best')
            out_path_aligned_all = Path(self.__output_dir, 'aligned', 'all')
    
            if not os.path.isdir(str(out_path_aligned_best)):
                os.makedirs(str(out_path_aligned_best))
            if not os.path.isdir(str(out_path_aligned_all)):
                os.makedirs(str(out_path_aligned_all))

        # iterate over the images
        with tqdm(total=len(self.__optical_filenames)) as pbar:
            for optical_name in self.__optical_filenames:
                # get the image paths
                index = optical_name.split('_')[0]
                optical_path = Path(self.__input_dir, optical_name)
                thermal_path = Path(self.__input_dir, index + '_thermal.png')
                thermal_raw_path = Path(self.__input_dir, index + '_thermal_raw.png')

                # load the images
                optical = cv2.imread(str(optical_path), -1)
                thermal = cv2.imread(str(thermal_path), -1)
                thermal_raw = cv2.imread(str(thermal_raw_path), -1)

                # align the images
                if self.__params['perspective']:
                    t_init = np.copy(self.__init_transform_perspective)
                else:
                    t_init = np.copy(self.__init_transform_affine)

                transformation = np.copy(t_init)

                if self.__params['alignment_method'] == 'mi':
                    (success, solver_fail, best_warped, transformation,
                        valid_warped) = self.align_images_mutual_information(optical,
                                                                             thermal,
                                                                             thermal_raw,
                                                                             t_init.copy())
                else:
                    raise ValueError('Unkown alignment method: ' + self.__params['alignment_method'])
                
                if success:
                    # update the averaged transformation
                    self.update_average_transformation(transformation)

                    if self.__params['save_aligned_images']:
                        cv2.imwrite(str(Path(out_path_aligned_best, index + '_optical.png')),
                                    best_warped)
                        cv2.imwrite(str(Path(out_path_aligned_best, index + '_thermal_raw.png')),
                                    thermal_raw)
                        cv2.imwrite(str(Path(out_path_aligned_best, index + '_thermal.png')),
                                    thermal)

                        for i, image in enumerate(valid_warped):
                            cv2.imwrite(str(Path(out_path_aligned_all, index + '_optical_' + str(i) + '.png')),
                                        image)
                else:
                    if solver_fail:
                        logging.warning('%s', optical_name)

                # update the progressbar
                pbar.update(1)

    def align_images_mutual_information(self, optical, thermal, thermal_raw, t_init):
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
    parser.add_argument('-y', '--yaml-config', default='create_dataset_toni/config/config_align_images.yaml', help='Yaml file containing the configs')
    parser.add_argument('-i', '--input-dir', default='tmp/processed/preprocessed', help='Input directory')
    parser.add_argument('-o', '--output-dir', default='/tmp/test', help='Output directory')

    args = parser.parse_args()

    worker = ImageAligner(args.input_dir, args.output_dir, args.yaml_config)

    worker.run()

if __name__ == "__main__":
    main()
