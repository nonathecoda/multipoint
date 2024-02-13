import argparse
import cv2
import math
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import yaml
from icecream import ic

## step 1: preprocess images
## step 2: match images based on pairs.txt file
class Preprocessor():
    '''
    Class to iterate over a bagfile and find image pairs based on the timestamp.
    '''

    def __init__(self, input_dir, output_dir, pair_file, config_file):
        ic("hi")
        self.__output_dir = output_dir
        self.__input_dir = input_dir
        self.__pair_file = pair_file

        with open(config_file, 'rt') as fh:
            self.__params = yaml.safe_load(fh)

    def run(self):

        if self.__params['save_preprocessed_images']:
            out_path_preprocessed = Path(self.__output_dir, 'preprocessed')

            if not os.path.isdir(str(out_path_preprocessed)):
                os.makedirs(str(out_path_preprocessed))

        pair_counter = 0
        # iterate over the pairs.txt file
        with open(self.__pair_file, 'r') as pair_file:
            cams = ["cam1", "cam2", "cam3", "cam4"]
            for row in pair_file:
                for cam in cams:
                    imagepair = row.strip().split(", ")
                    imagepair[0] = cam + "/" + imagepair[0].replace("cam1", cam)
                    imagepair[1] = "camT/" + imagepair[1]

                    optical_path = ''.join((self.__input_dir, imagepair[0]))
                    thermal_path = ''.join((self.__input_dir, imagepair[1]))

                    optical_image, thermal_image, thermal_image_rescaled = self.preprocess_images(optical_path, thermal_path)

                    if self.__params['save_preprocessed_images']:
                        cv2.imwrite(str(Path(out_path_preprocessed, str(pair_counter) + '_optical.png')),
                                    optical_image)
                        cv2.imwrite(str(Path(out_path_preprocessed, str(pair_counter) + '_thermal_raw.png')),
                                    thermal_image)
                        cv2.imwrite(str(Path(out_path_preprocessed, str(pair_counter) + '_thermal.png')),
                                    (thermal_image_rescaled * 65535).astype('uint16'))
                    
                    pair_counter += 1
    
    def preprocess_images(self, optical_path, thermal_path):

        # extract the images from the messages
        cv_optical = cv2.imread(optical_path, 0)
        cv_thermal = cv2.imread(thermal_path, 0)

        # downscale the optical if desired
        if self.__params['image/optical/downscale']:
            ratio = float(cv_thermal.shape[0])/cv_optical.shape[0]
            cv_optical = cv2.resize(cv_optical,(int(cv_optical.shape[1]*ratio), int(cv_thermal.shape[0])))

        # convert thermal image to a floating point image and rescale it
        if self.__params['image/thermal/rescale_outlier_rejection']:
            lower_bound = np.percentile(cv_thermal, 1)
            upper_bound = np.percentile(cv_thermal, 99)

            cv_thermal_rescaled = cv_thermal
            cv_thermal_rescaled[cv_thermal_rescaled < lower_bound] = lower_bound
            cv_thermal_rescaled[cv_thermal_rescaled > upper_bound] = upper_bound

        cv_thermal_rescaled = cv2.normalize(cv_thermal_rescaled, None, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        return cv_optical, cv_thermal, cv_thermal_rescaled

def main():
    parser = argparse.ArgumentParser(description='Extract images from a rosbag and save them as pairs')
    parser.add_argument('-y', '--yaml-config', default='arss_scripts/image_alignment/configs/config_extract_images.yaml', help='Yaml file containing the configs')
    parser.add_argument('-p', '--pair-file', default='data/data_arss/datapairs_best.txt', help='Txt file with name of the pairs')
    parser.add_argument('-i', '--input-dir', default='/Users/antonia/dev/UNITN/remote_sensing_systems/data/ARSS_P3/', help='Input directory to hdf5 files')
    parser.add_argument('-o', '--output-dir', default='/Users/antonia/dev/UNITN/remote_sensing_systems/multipoint/tmp/processed', help='Output directory')

    args = parser.parse_args()

    worker = Preprocessor(args.input_dir, args.output_dir, args.pair_file, args.yaml_config)

    worker.run()

if __name__ == "__main__":
    main()
