from icecream import ic
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize
import os
import h5py
import atexit

counter = 0

def cleanup(training_file):
    training_file.close()
    print("Closed training.hdf5 before exiting.")

def remove_duplicates(path_to_image_pairs):
    unique_lines = set()
    ordered_lines = []

    # Read the file and keep the order, skipping duplicates
    with open(path_to_image_pairs, 'r') as file:
        for line in file:
            stripped_line = line.strip()
            if stripped_line not in unique_lines:
                unique_lines.add(stripped_line)
                ordered_lines.append(line)  # Preserve the original line, including whitespace

    # Write the ordered lines back to the file
    with open(path_to_image_pairs, 'w') as file:
        file.writelines(ordered_lines)

def get_warped_other_cam(cam_index, H, width, height):
    cami_path = "/Users/antonia/dev/UNITN/remote_sensing_systems/data/ARSS_P3/cam"+str(cam_index)+"/" + imagepair[0].replace("cam1_", "cam"+str(cam_index)+"_")
    img_optical_cami = cv2.imread(cami_path, 0)
    img_optical_cami = cv2.resize(img_optical_cami, (312, 232))
    img_optical_cami = img_optical_cami/img_optical_cami.max()
    warped_img_optical_cami = cv2.warpPerspective(img_optical_cami, h_finder.H_final, dsize=(h_finder.img_thermal.shape[1], h_finder.img_thermal.shape[0]), flags=cv2.INTER_CUBIC)
    return warped_img_optical_cami

class Homography_Finder:

    def __init__(self, imagepair):
        self.restart = True
        self.skip = False

        self.keypoints_optical = []
        self.keypoints_thermal = []

        self.warped_optical = None
        self.img_optical = None
        self.img_thermal = None

        self.H_manual = None
        self.H_optimized_0 = None
        self.H_optimized_1 = None
        self.H_optimized_2 = None

        self.H_final = None
        self.final_warped_optical = None

        # Load the images
        cam1 = "/Users/antonia/dev/UNITN/remote_sensing_systems/data/ARSS_P3/cam1/" + imagepair[0]
        img_optical = cv2.imread(cam1, 0)
        camT = "/Users/antonia/dev/UNITN/remote_sensing_systems/data/ARSS_P3/camT/" + imagepair[1]
        img_thermal = cv2.imread(camT, 0)

        #resize images to 233, 314
        img_optical = cv2.resize(img_optical, (312, 232))
        img_thermal = cv2.resize(img_thermal, (312, 232))

        #normalize thermal images (you could also use np.clip)
        img_thermal = np.clip(img_thermal, np.percentile(img_thermal, 1), np.percentile(img_thermal, 99))
        self.img_thermal = img_thermal/img_thermal.max()
        # normalize optical images
        self.img_optical = img_optical/img_optical.max()
        
        if img_optical.shape != img_thermal.shape:
            raise ValueError("Images must have the same dimensions.")
        self.display_img = np.hstack((self.img_optical, self.img_thermal))

        self.x_offset = self.display_img.shape[1]/2

    def show_img(self):
        window_name = str(counter) + ' / 277. Find 4 keypoint pairs, then press "d"'
        # Create a window and set the mouse callback
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self.mouse_callback, param=window_name)
        
        # Display the images initially
        while(1):
            cv2.resizeWindow(window_name, 1400, 500)
            cv2.imshow(window_name, self.display_img)
            if cv2.waitKey(20) & 0xFF == 100:
                break
            elif cv2.waitKey(20) & 0xFF == 27:
                cv2.destroyAllWindows()
                exit()
        cv2.destroyAllWindows()

        # Print the manually selected keypoints for each image
        print("Manually selected keypoints for Image 1:")
        for i, point in enumerate(self.keypoints_optical, 1):
            print(f"Keypoint {i}: ({point[0]}, {point[1]})")

        print("\nManually selected keypoints for Image 2:")
        for i, point in enumerate(self.keypoints_thermal, 1):
            print(f"Keypoint {i}: ({point[0]}, {point[1]})")
        
    def mouse_callback(self, event, x, y, flags, params):
            if event == cv2.EVENT_LBUTTONDOWN:
                cv2.circle(self.display_img, (x, y), 3, (0, 255, 0), -1)
                if x >= self.x_offset:
                    self.keypoints_thermal.append((x-self.x_offset, y))
                else:
                    self.keypoints_optical.append((x, y))
                # Display the images with keypoints
                cv2.resizeWindow(params, 1400, 500)
                cv2.imshow(params, self.display_img)

    def calc_manual_H(self):
        try:
            self.H_manual = cv2.findHomography(np.array(self.keypoints_optical), np.array(self.keypoints_thermal), cv2.RANSAC)[0]
            self.warped_optical = cv2.warpPerspective(self.img_optical, self.H_manual, dsize=(self.img_thermal.shape[1], self.img_thermal.shape[0]), flags=cv2.INTER_CUBIC)
        except cv2.error:
            self.restart = True
            print("Not 4 pairs of keypoints selected. Please restart.")
            
    def optimize_homography(self, bins):
        def cost_function(H):
            return -self.compute_mi_score(bins, self.warped_optical, self.img_thermal, H)
        
        # Perform optimization
        H = self.H_manual
        res = minimize(cost_function, H, method='nelder-mead',options={'xatol': 1e-10, 'disp': True, 'maxiter': 1000})
        
        return res.x.reshape(3,3)

    def compute_mi_score(self, bins, img1, img2, H):
        """Computes the mutual information (MI) score between two images warped according to a homography H."""
        warped_img1 = cv2.warpPerspective(img1, H.reshape(3,3), dsize=(img2.shape[1], img2.shape[0]), flags=cv2.INTER_CUBIC)
        # Compute histograms of the images
        hist_2d, x_edges, y_edges = np.histogram2d(warped_img1.ravel(), img2.ravel(),bins=bins)
        #compute MI score
        mi_score = self.mutual_information(hist_2d)
        return mi_score
    
    def mutual_information(self, hgram):
        """ Mutual information for joint histogram
        """
        # Convert bins counts to probability values
        pxy = hgram / float(np.sum(hgram))
        px = np.sum(pxy, axis=1) # marginal for x over y
        py = np.sum(pxy, axis=0) # marginal for y over x
        px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
        # Now we can do the calculation using the pxy, px_py 2D arrays
        nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
        return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))
        
    def choose_homography(self):

        def choose_homography_key_event(event):
            if event.key == 'q':
                plt.close()  # Close the window when 'q' is pressed
            elif event.key == '1':
                self.H_final = self.H_manual
                self.final_warped_optical = warped_optical_m
                ic(self.final_warped_optical is None)
                print("Chose manual homography")
                plt.close()  
            elif event.key == 'r':
                print("restart")
                plt.close()
                self.restart = True
            elif event.key == 'n':
                print("skip image")
                plt.close()
                self.skip = True
            '''
            elif event.key == '2':
                self.H_final = self.H_optimized_0
                self.final_warped_optical = warped_optical_o_0
                print("Chose optimized homography 0")
                plt.close()  
            elif event.key == '3':
                self.H_final = self.H_optimized_1
                self.final_warped_optical = warped_optical_o_1
                print("Chose optimized homography 0")
                plt.close()  
            elif event.key == '4':
                self.H_final = self.H_optimized_2
                self.final_warped_optical = warped_optical_o_2
                print("Chose optimized homography 0")
                plt.close() 
            '''
            
        
        # manual homography
        warped_optical_m = cv2.warpPerspective(self.img_optical, self.H_manual, dsize=(self.img_thermal.shape[1], self.img_thermal.shape[0]), flags=cv2.INTER_CUBIC)
        # plot different homographies
        '''
        f, axarr = plt.subplots(1,4, figsize=(20, 10))
        axarr[0].imshow(self.img_thermal, cmap='gray')
        axarr[0].imshow(warped_optical_m, cmap='gray', alpha=0.5)
        axarr[0].set_title('Manual H - Key 1')
        '''

        plt.imshow(self.img_thermal, cmap='gray')
        plt.imshow(warped_optical_m, cmap='gray', alpha=0.5)
        plt.title(str(counter) + ' / 227. 1: confirm, r: restart. n: next.')
        try:
            # optimized homography
            warped_optical_o_0 = cv2.warpPerspective(self.warped_optical, self.H_optimized_0, dsize=(self.img_thermal.shape[1], self.img_thermal.shape[0]), flags=cv2.INTER_CUBIC)
            #... other homographies
            warped_optical_o_1 = cv2.warpPerspective(self.warped_optical, self.H_optimized_1, dsize=(self.img_thermal.shape[1], self.img_thermal.shape[0]), flags=cv2.INTER_CUBIC)
            warped_optical_o_2 = cv2.warpPerspective(self.warped_optical, self.H_optimized_2, dsize=(self.img_thermal.shape[1], self.img_thermal.shape[0]), flags=cv2.INTER_CUBIC)
            
            axarr[1].imshow(self.img_thermal, cmap='gray')
            axarr[1].imshow(warped_optical_o_0, cmap='gray', alpha=0.5)
            axarr[1].set_title('Optimized H 0 - Key 2')
            axarr[2].imshow(self.img_thermal, cmap='gray')
            axarr[2].imshow(warped_optical_o_1, cmap='gray', alpha=0.5)
            axarr[2].set_title('Optimized H 1 - Key 3')
            axarr[3].imshow(self.img_thermal, cmap='gray')
            axarr[3].imshow(warped_optical_o_2, cmap='gray', alpha=0.5)
            axarr[3].set_title('Optimized H 2 - Key 4')
        except cv2.error:
            warped_optical_o_0 = None
            warped_optical_o_1 = None
            warped_optical_o_2 = None
            print("No optimization.")
           
        plt.connect('key_press_event', choose_homography_key_event)
        plt.show()
    
    def reset(self):
        self.restart = False
        self.skip = False
        self.keypoints_optical = []
        self.keypoints_thermal = []
        self.H_manual = None
        self.H_optimized_0 = None
        self.H_optimized_1 = None
        self.H_optimized_2 = None
        self.H_final = None
        self.warped_optical = None
        self.final_warped_optical = None
        self.display_img = np.hstack((self.img_optical, self.img_thermal))

if __name__ == "__main__":
    
    user_input = input("Do you want to execute the code? All hdf5 files will be overwritten. (yes/no): ")

    # Check the user's response
    if user_input.lower() == 'yes':
        print("Executing the code...")
        # Place your code here that you want to execute
    else:
        print("Execution cancelled.")
        exit()

    path_training_file = "/Users/antonia/dev/UNITN/remote_sensing_systems/multipoint/tmp/processed/method_5/aligned/training.hdf5"
    if os.path.exists(path_training_file):
        # Delete the file
        os.remove(path_training_file)
    training_file = h5py.File(path_training_file, 'w')

    atexit.register(cleanup, training_file)

    '''
    Find homographies manually for each image pair.
    Image pair: map each thermal to each optica
    
    '''
    path_to_data = "/Users/antonia/dev/UNITN/remote_sensing_systems/data/ARSS_P3"
    path_to_image_pairs = "/Users/antonia/dev/UNITN/remote_sensing_systems/arss_aerial_feature_matching/groundtruth_gui/data/pairs/datapairs.txt"

    remove_duplicates(path_to_image_pairs)

    # TODO: start with pairs cam1  - T, then replace cam1 with cam2, cam3, cam4
    image_pair_file = open(path_to_image_pairs, 'r')

    for line in image_pair_file:
        imagepair = line.strip().split(", ")
        h_finder = Homography_Finder(imagepair)
        while h_finder.restart == True:
            h_finder.reset()
            h_finder.show_img()
            h_finder.calc_manual_H()
            if h_finder.restart == True:
                continue
            bins_0 = [312, 156] #(312, 232))
            bins_1 = [200, 100]
            bins_2 = [46, 23]
            #h_finder.H_optimized_0 = h_finder.optimize_homography(bins_0)
            #h_finder.H_optimized_1 = h_finder.optimize_homography(bins_1)
            #h_finder.H_optimized_2 = h_finder.optimize_homography(bins_2)
            h_finder.choose_homography()
            if h_finder.skip == True:
                break
        if h_finder.skip == True:
                h_finder.skip = False
                counter += 1
                continue

        #plt.imshow(h_finder.final_warped_optical, cmap='gray')
        #plt.show()

        #CAM 1: save warped optical and thermal image to "training.hdf5"
        group = training_file.create_group(str(counter) + "_cam1")
        dataset = group.create_dataset("optical", data=np.array(h_finder.final_warped_optical))
        dataset = group.create_dataset("thermal", data=np.array(h_finder.img_thermal))

        #CAM 2: save warped optical and thermal image to "training.hdf5"
        warped_img_optical_cam2 = get_warped_other_cam(2, h_finder.H_final, h_finder.img_thermal.shape[1], h_finder.img_thermal.shape[0])
        group = training_file.create_group(str(counter) + "_cam2")
        dataset = group.create_dataset("optical", data=np.array(warped_img_optical_cam2))
        dataset = group.create_dataset("thermal", data=np.array(h_finder.img_thermal))

        #CAM 3: save warped optical and thermal image to "training.hdf5"
        warped_img_optical_cam3 = get_warped_other_cam(3, h_finder.H_final, h_finder.img_thermal.shape[1], h_finder.img_thermal.shape[0])
        group = training_file.create_group(str(counter) + "_cam3")
        dataset = group.create_dataset("optical", data=np.array(warped_img_optical_cam3))
        dataset = group.create_dataset("thermal", data=np.array(h_finder.img_thermal))

        #CAM 4: save warped optical and thermal image to "training.hdf5"
        warped_img_optical_cam4 = get_warped_other_cam(4, h_finder.H_final, h_finder.img_thermal.shape[1], h_finder.img_thermal.shape[0])
        group = training_file.create_group(str(counter) + "_cam4")
        dataset = group.create_dataset("optical", data=np.array(warped_img_optical_cam4))
        dataset = group.create_dataset("thermal", data=np.array(h_finder.img_thermal))

        counter += 1