import torch.nn as nn

class YourModelClass(nn.Module):
    """
    A sample neural network model for predicting future VIL images based on input frames.

    Parameters
    ----------
    input_channels : int
        Number of input channels (e.g., number of input image types).
    output_channels : int
        Number of output channels (e.g., number of target image types).

    Attributes
    ----------
    conv1 : torch.nn.Conv2d
        First convolutional layer.
    relu : torch.nn.ReLU
        ReLU activation function.
    conv2 : torch.nn.Conv2d
        Second convolutional layer.

    Methods
    -------
    forward(x)
        Defines the forward pass of the network.
    """

    def __init__(self, input_channels, output_channels):
        super(YourModelClass, self).__init__()
        # Define your model layers here
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(64, output_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        """
        Defines the forward pass of the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (batch_size, input_channels, height, width).

        Returns
        -------
        torch.Tensor
            Output tensor with shape (batch_size, output_channels, height, width).
        """
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return x
