import h5py


def compute_frame_mean_std(h5_path, img_types):
    """
    Computes the global mean and standard deviation for each image type in an HDF5 dataset.
    
    This function iterates over all event groups in the HDF5 file and calculates the 
    mean and standard deviation for each image type across all events.
    
    Parameters:
    ----------
    h5_path : str
        Path to the HDF5 file containing the dataset.
    img_types : list of str
        List of image types (datasets) for which the mean and standard deviation should be computed.

    Returns:
    -------
    dict
        A dictionary where keys are image types and values are tuples (mean, std).
    
    Example:
    --------
    >>> img_types = ['vis', 'ir069', 'ir107', 'vil']
    >>> global_mean_std_dic = compute_frame_mean_std('data/train.h5', img_types)
    >>> print(global_mean_std_dic)
    {'vis': (mean_value, std_value), 'ir069': (mean_value, std_value), ...}
    """
    global_mean_std_dic = {}

    with h5py.File(h5_path, "r") as f:
        for img in img_types:
            mean_sum = 0
            std_sum = 0
            num_events = len(f.keys())

            for event_id in f.keys():  # Iterate over event groups
                data = f[event_id][img][:]
                mean_sum += data.mean()
                std_sum += data.std()

            # Compute the global mean and standard deviation
            global_mean = mean_sum / num_events if num_events > 0 else 0
            global_std = std_sum / num_events if num_events > 0 else 0

            # Store in dictionary
            global_mean_std_dic[img] = (global_mean, global_std)

    return global_mean_std_dic

# Example usage
# img_types = ['vis', 'ir069', 'ir107', 'vil']
# global_mean_std_dic = compute_frame_mean_std('data/train.h5', img_types)
# Computing time can be high; finalised version below. 
global_mean_std_dic = {'vis': (1771.437, 1099.165),
                       'ir069': (-3820.202, 998.747),
                       'ir107': (-1663.359, 2226.835),
                       'vil': (24.554, 37.624)}


def compute_frame_min_max(h5_path, img_types):
    """
    Computes the global minimum and maximum values for specified image types across all event groups in an HDF5 file.

    Parameters:
    -----------
    h5_path : str
        Path to the HDF5 file containing the dataset.
    img_types : list of str
        List of image types to compute minimum and maximum values for.

    Returns:
    --------
    dict
        A dictionary with image types as keys and a tuple of (min, max) as values.

    Raises:
    -------
    FileNotFoundError
        If the HDF5 file does not exist at the specified path.
    """
    dic_min_max = {}
    with h5py.File(h5_path, "r") as f:
        for img in img_types:
            global_min = float('inf')
            global_max = float('-inf')
            for event_id in f.keys():  # Iterate over event groups
                data = f[event_id][img][:]
                data_min = data.min()
                data_max = data.max()
                if data_min < global_min:
                    global_min = data_min
                if data_max > global_max:
                    global_max = data_max
            dic_min_max[img] = (global_min, global_max)
    return dic_min_max
