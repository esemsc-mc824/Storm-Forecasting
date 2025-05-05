def normalise(input_image, img_type, normalization_type='standard_normal', global_min_max_dic=None, global_mean_std_dic=None):
    """
    Normalizes the input image using either min-max normalization or standard normalization.

    Parameters
    ----------
    input_image : numpy.ndarray or torch.Tensor
        The image data to be normalized.
    img_type : str
        The type of the image. Must be one of ['vil', 'ir069', 'ir107', 'vis'].
    normalization_type : str, optional
        The type of normalization to apply. 
        - 'min_max' for min-max normalization.
        - 'standard_normal' for standard score normalization.
        Default is 'standard_normal'.
    global_min_max_dic : dict, optional
        A dictionary containing the global minimum and maximum values for each image type.
        Required if `normalization_type` is 'min_max'.
    global_mean_std_dic : dict, optional
        A dictionary containing the global mean and standard deviation for each image type.
        Required if `normalization_type` is 'standard_normal'.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        The normalized image.

    Raises
    ------
    ValueError
        If `input_image` is None.
        If `img_type` is not among the supported types.
        If required normalization dictionaries are not provided.
        If an unsupported normalization type is specified.
    """
    if input_image is None:
        raise ValueError("Please provide the data")
    if img_type not in ['vil', 'ir069', 'ir107', 'vis']:
        raise ValueError("Invalid image type. Possible types: ['vil', 'ir069', 'ir107', 'vis']")
  
    if normalization_type == 'min_max':
        if global_min_max_dic is None:
            raise ValueError("global_min_max_dic must be provided for min_max normalization")
        min_val, max_val = global_min_max_dic[img_type]
        return (input_image - min_val) / (max_val - min_val)
    
    elif normalization_type == 'standard_normal':
        if global_mean_std_dic is None:
            raise ValueError("global_mean_std_dic must be provided for standard_normal normalization")
        mean, std = global_mean_std_dic[img_type]
        return (input_image - mean) / std
    
    else:
        raise ValueError("Unsupported normalization type.")
