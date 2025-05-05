import unittest
import numpy as np
import torch
from DataProcessing import (
    normalise,
)
from DataProcessing.models import YourModelClass

class TestPackage(unittest.TestCase):
    def setUp(self):
        """
        Set up test fixtures. This method is called before each test.
        """
        # Mock data for normalization tests
        self.input_image_min_max = np.array([0, 128, 255])
        self.min_max_dic = {'vil': (0, 255)}
        self.input_image_standard = np.array([64, 128, 192])
        self.mean_std_dic = {'vil': (128.0, 64.0)}
        
    def test_normalise_min_max(self):
        """
        Test the normalise function with min-max normalization.
        """
        normalised = normalise(
            input_image=self.input_image_min_max,
            img_type='vil',
            normalization_type='min_max',
            global_min_max_dic=self.min_max_dic
        )
        expected = np.array([0.0, 128/255, 1.0])
        np.testing.assert_almost_equal(normalised, expected)

    def test_normalise_standard_normal(self):
        """
        Test the normalise function with standard normal normalization.
        """
        normalised = normalise(
            input_image=self.input_image_standard,
            img_type='vil',
            normalization_type='standard_normal',
            global_mean_std_dic=self.mean_std_dic
        )
        expected = np.array([-1.0, 0.0, 1.0])
        np.testing.assert_almost_equal(normalised, expected)

    def test_YourModelClass(self):
        """
        Test the YourModelClass neural network.
        """
        input_channels = 1
        output_channels = 1
        model = YourModelClass(input_channels, output_channels)
        self.assertIsInstance(model, YourModelClass)
        # Create a dummy input tensor
        dummy_input = torch.randn(4, input_channels, 64, 64)  # batch_size=4, channels=1, height=64, width=64
        output = model(dummy_input)
        self.assertEqual(output.shape, (4, output_channels, 64, 64))

    def test_invalid_normalise_type(self):
        """
        Test the normalise function with an invalid normalization type.
        """
        with self.assertRaises(ValueError):
            normalise(
                input_image=self.input_image_min_max,
                img_type='vil',
                normalization_type='invalid_type',
                global_min_max_dic=self.min_max_dic
            )

    def test_invalid_image_type_normalise(self):
        """
        Test the normalise function with an invalid image type.
        """
        with self.assertRaises(ValueError):
            normalise(
                input_image=self.input_image_min_max,
                img_type='invalid_type',
                normalization_type='min_max',
                global_min_max_dic=self.min_max_dic
            )

    def test_normalise_missing_dic_min_max(self):
        """
        Test the normalise function with min-max normalization without providing the required dictionary.
        """
        with self.assertRaises(ValueError):
            normalise(
                input_image=self.input_image_min_max,
                img_type='vil',
                normalization_type='min_max',
                global_min_max_dic=None
            )

    def test_normalise_missing_dic_standard_normal(self):
        """
        Test the normalise function with standard normal normalization without providing the required dictionary.
        """
        with self.assertRaises(ValueError):
            normalise(
                input_image=self.input_image_standard,
                img_type='vil',
                normalization_type='standard_normal',
                global_mean_std_dic=None
            )

if __name__ == '__main__':
    unittest.main()
