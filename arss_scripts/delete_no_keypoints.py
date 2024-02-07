import h5py
from icecream import ic

path_training = '/Users/antonia/dev/UNITN/remote_sensing_systems/arss_aerial_feature_matching/groundtruth_gui/data/training_del.hdf5'
path_labels = '/Users/antonia/dev/UNITN/remote_sensing_systems/multipoint/tmp/labels.hdf5'

# Open both files
with h5py.File(path_training, 'r+') as file1, h5py.File(path_labels, 'r') as file2:
    # Create a list of keys in label file for comparison
    keys_in_file2 = list(file2.keys())
    
    # Create a list of keys to delete from training file (to avoid modifying the dict while iterating)
    keys_to_delete = [key for key in file1.keys() if key not in keys_in_file2]

    # Delete the non-matching groups from the training file
    for key in keys_to_delete:
        del file1[key]
        print(f"Deleted group '{key}' from training file.")