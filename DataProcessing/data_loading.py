import h5py
import torch
from torch.utils.data import Dataset
from .normalisation import normalise

class DataGenerator(Dataset):
    """
    A PyTorch Dataset class for loading and preprocessing data from an HDF5 file.

    Parameters
    ----------
    input_img_types : list of str
        List of input image types (e.g., ['vil', 'vis']).
    target_img_type : str, optional
        The target image type to predict (default is 'vil').
    h5_path : str, optional
        Path to the HDF5 file containing the dataset (default is 'data/train.h5').
    shuffle : bool, optional
        Whether to shuffle the data (default is True).
    max_frames : int, optional
        Maximum number of frames per tensor (default is 36).
    stack : bool, optional
        Whether to stack the input images (default is True).
    scaling_shape : tuple, optional
        Shape to rescale the images (default is (192, 192)).
    scaling_mode : str, optional
        Mode of rescaling (default is 'bilinear').
    target_out : bool, optional
        Whether the target images should be within the time range of the input images (default is False).
    T_in : int, optional
        Number of input time frames (default is 12).
    T_out : int, optional
        Number of output time frames (default is 12).
    step : int, optional
        Step size for sliding windows (default is 6).

    Attributes
    ----------
    input_img_types : list of str
        List of input image types.
    target_img_type : str
        Target image type.
    h5_path : str
        Path to the HDF5 file.
    max_frames : int
        Maximum number of frames per tensor.
    shuffle : bool
        Whether to shuffle the data.
    stack : bool
        Whether to stack the input images.
    scaling_shape : tuple
        Shape for rescaling the images.
    scaling_mode : str
        Mode of rescaling.
    target_out : bool
        Whether the target images should be within the time range of the input images.
    T_in : int
        Number of input time frames.
    T_out : int
        Number of output time frames.
    step : int
        Step size for sliding windows.
    samples : list of tuples
        List of (event_id, start_frame) tuples representing samples.

    Methods
    -------
    __len__()
        Returns the total number of samples.
    __getitem__(idx)
        Retrieves the sample at the specified index.

    Example Usage:
    --------------
    1a) Training dataset for input set A (no scaling)
    train_dataset_1a = DataGenerator(
        input_img_types=img_types_a, 
        event_ids=train_ids, 
        target_out=True, 
        scaling=False
    )
    
    1b) Testing dataset for input set A (no scaling)
    train_dataset_1b = DataGenerator(
        input_img_types=img_types_a, 
        event_ids=test_ids, 
        target_out=True, 
        scaling=False
    )
    
    2) Training dataset for input set B with scaling to (384, 384)
    train_dataset_2 = DataGenerator(
        input_img_types=img_types_2, 
        event_ids=train_ids, 
        target_out=False, 
        scaling=True, 
        scaling_shape=(384, 384)
     )
     Expected Output Shapes:
    ------------------------
    INPUT.shape  = (batch_size, num_input_frames, time_steps, height, width)
    TARGET.shape = (batch_size, time_steps, height, width)
    """

    def __init__(
        self,
        input_img_types,
        event_ids,
        scaling = False,
        target_img_type='vil',
        h5_path='data/train.h5',
        factor=0.5,
        shuffle=True,
        max_frames=36,
        stack=True,
        scaling_shape=(192, 192),
        scaling_mode='bilinear',
        target_out=False,
        T_in=12,
        T_out=12,
        step=6,
    ):
        self.input_img_types = input_img_types  # List of image type strings
        self.event_ids = event_ids
        self.target_img_type = target_img_type
        self.h5_path = h5_path  # Path to the HDF5 file
        self.target_img_type = target_img_type  # Target image type string
        self.factor = factor  # Fraction of dataset to use
        self.max_frames = max_frames  # Number of frames per tensor
        self.shuffle = shuffle  # Whether to shuffle data
        self.stack = stack #stack
        self.scaling = scaling
        self.scaling_shape = scaling_shape #shape for upsample
        self.scaling_mode = scaling_mode #mode of upsampling
        self.target_out = target_out # wehter the target_images should within the time range of the input_images
        self.T_in = T_in #number of time frames in
        self.T_out = T_out #number of time frames out
        self.samples = []
        self.step = step #number of step for sliding windows


        if self.shuffle:
            torch.manual_seed(42)
            shuffled_indices = torch.randperm(len(self.event_ids)).tolist()
            self.event_ids = [self.event_ids[i] for i in shuffled_indices]

        if isinstance(self.T_in, int) and isinstance(self.T_out, int):
            if self.T_in + self.T_out > self.max_frames:
                raise ValueError("The sum of T_in and T_out cannot exceed max_frames.")

        if self.target_out:
            with h5py.File(self.h5_path, 'r') as f:
                for event_id in self.event_ids:
                    T = 36
                    if T < 24:
                        continue
                    for start in range(0, T - (self.T_in + self.T_out) + 1, self.step):
                        self.samples.append((event_id, start))

    def __len__(self):
        """
        Returns the total number of event_ids.

        Returns
        -------
        int
            Number of samples in the dataset.
        """
        return len(self.event_ids)

    def __getitem__(self, idx):
        """
        Retrieves the sample at the specified index.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        tuple of torch.Tensor
            A tuple containing:
            - input_frames: Tensor of input images.
            - target_frame: Tensor of target images.
        """
        with h5py.File(self.h5_path, 'r') as f:
            input_frames = {}
            if self.target_out:
                event_id, start = self.samples[idx]
                target_frame = torch.tensor(f[event_id][self.target_img_type][..., start+self.T_in: start+self.T_out+self.T_in].astype(np.float32))
            else:
                event_id = self.event_ids[idx]
                target_frame = torch.tensor(f[event_id][self.target_img_type][..., :self.max_frames].astype(np.float32))

            for img in self.input_img_types:
                if img not in ['vis', 'ir069', 'ir107', 'vil']:
                    raise ValueError(f"Input image {img} not found. Please use one of ['vis', 'ir069', 'ir107', 'vil'].")

                if self.target_out:
                    frame = torch.tensor(f[event_id][img][..., start:start+self.T_in].astype(np.float32))
                else:
                    frame = torch.tensor(f[event_id][img][..., :self.max_frames].astype(np.float32))

                frame = normalize(frame, input_image_type=img)
                if self.scaling == True:
                  frame = F.interpolate(frame.permute(2, 0, 1).unsqueeze(1), size=self.scaling_shape, mode=self.scaling_mode, align_corners = False).squeeze()
                input_frames[img] = frame

            target_frame = normalize(target_frame, input_image_type=self.target_img_type)
            if self.scaling == True:
                target_frame = F.interpolate(target_frame.permute(2, 0, 1).unsqueeze(1), size=self.scaling_shape, mode=self.scaling_mode).squeeze()

            return torch.stack(tuple(input_frames.values())), target_frame


def load_event(event_id, file_path="data/train.h5"):
    """
    Loads image data for a specific event into a dictionary.

    Parameters
    ----------
    event_id : str
        The ID of the event to load.
    file_path : str, optional
        Path to the HDF5 dataset file (default is 'data/train.h5').

    Returns
    -------
    dict
        A dictionary where keys are image types and values are the corresponding image arrays.

    Raises
    ------
    KeyError
        If the event_id is not found in the HDF5 file.
    """
    with h5py.File(file_path, 'r') as f:
        if event_id not in f:
            raise KeyError(f"Event ID {event_id} not found in the dataset.")
        event = {img_type: f[event_id][img_type][:] for img_type in ['vis', 'ir069', 'ir107', 'vil', 'lght']}
    return event

def load_multiple_events(event_ids, file_path="data/train.h5"):
    """
    Loads lightning data for multiple events into a dictionary.

    Parameters
    ----------
    event_ids : list of str
        List of event IDs to load.
    file_path : str, optional
        Path to the HDF5 dataset file (default is 'data/train.h5').

    Returns
    -------
    dict
        A dictionary where keys are event IDs and values are their lightning data as pandas DataFrames.

    Raises
    ------
    KeyError
        If any of the event_ids are not found in the HDF5 file.
    """
    import pandas as pd

    events_data = {}
    with h5py.File(file_path, "r") as f:
        for event_id in event_ids:
            if event_id in f:
                if "lght" in f[event_id]:
                    lightning_data = f[event_id]["lght"][:]
                    events_data[event_id] = pd.DataFrame(
                        lightning_data,
                        columns=["t", "lat (deg)", "lon (deg)", "vil pixel x", "vil pixel y"]
                    )
                else:
                    print(f"Warning: No lightning data for event {event_id}")
            else:
                print(f"Warning: Event ID {event_id} not found in dataset")
    return events_data
