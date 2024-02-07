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
                optical = cv2.imread(str(optical_path), -1)
                thermal = cv2.imread(str(thermal_path), -1)

                #resize optical to thermal
                optical = cv2.resize(optical, (thermal.shape[1], thermal.shape[0]))

                #apply gaussian Blur to remove the Gaussian Noise from the image
                optical_blurr = cv2.GaussianBlur(optical,(5,5),0)
                edges_optical = cv2.Laplacian(optical_blurr, -1, ksize=5, scale=1,delta=0,borderType=cv2.BORDER_DEFAULT)

                thermal_blurr = cv2.GaussianBlur(thermal,(7,7),0)
                edges_thermal = cv2.Laplacian(thermal_blurr, -1, ksize=5, scale=1,delta=0,borderType=cv2.BORDER_DEFAULT)

                #show with matplotlib
                fig, ax = plt.subplots(2,2, figsize=(10,5))
                ax[0,0].imshow(optical, cmap='gray')
                ax[0,0].set_title('Optical Image (with Gaussian blur)')
                ax[0,1].imshow(edges_optical, cmap='gray')
                ax[0,1].set_title('Edges Optical Image')
                ax[1,0].imshow(thermal, cmap='gray')
                ax[1,0].set_title('Thermal Image (with Gaussian blur)')
                ax[1,1].imshow(edges_thermal, cmap='gray')
                ax[1,1].set_title('Edges Thermal Image')
                plt.show()


                exit()
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