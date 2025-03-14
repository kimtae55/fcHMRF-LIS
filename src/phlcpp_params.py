"""
MIT License

Copyright (c) 2019 Sadeep Jayasumana

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import torch
import random
from src import config

class DenseCRFParams(object):
    """
    Parameters for the DenseCRF model
    """

    def __init__(
        self,
        image=None,
        alpha=160.0,
        beta=3.0,
        gamma=3.0,
        spatial_ker_weight=3.0,
        bilateral_ker_weight=5.0,
    ):
        """
        Default values were taken from https://github.com/sadeepj/crfasrnn_keras. More details about these parameters
        can be found in https://arxiv.org/pdf/1210.5644.pdf

        Args:
            alpha:                  Bandwidth for the spatial component of the bilateral filter
            beta:                   Bandwidth for the color component of the bilateral filter
            gamma:                  Bandwidth for the spatial filter
            spatial_ker_weight:     Spatial kernel weight
            bilateral_ker_weight:   Bilateral kernel weight
        """
        if image is not None:
            self._initialize_sigmas(image) 
        else:
            self.alpha = alpha
            self.beta = beta
            self.gamma = gamma
            self.spatial_ker_weight = spatial_ker_weight
            self.bilateral_ker_weight = bilateral_ker_weight

    def _compute_std_spatial(self, image):
        image_shape = image.shape
        # Generate a 3D grid of positions for the image
        x = torch.arange(image_shape[-3])
        y = torch.arange(image_shape[-2])
        z = torch.arange(image_shape[-1])
        # Create a meshgrid of coordinates for the entire 3D image
        X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
        flat_X = X.flatten().float()  
        flat_Y = Y.flatten().float()  
        flat_Z = Z.flatten().float()  
        diff_x = flat_X[:, None] - flat_X[None, :]  # Pairwise differences in x direction
        diff_y = flat_Y[:, None] - flat_Y[None, :]  # Pairwise differences in y direction
        diff_z = flat_Z[:, None] - flat_Z[None, :]  # Pairwise differences in z direction
        # Compute the standard deviation of the pairwise differences along each axis
        sigma_x = torch.std(diff_x)
        sigma_y = torch.std(diff_y)
        sigma_z = torch.std(diff_z)
        # Return the sigma values for each axis
        return (sigma_x, sigma_y, sigma_z)

    def compute_std_spatial_sampled(self, image, sample_fraction=0.5):
        image_shape = image.shape
        # Generate a 3D grid of positions for the image
        x = torch.arange(image_shape[-3])
        y = torch.arange(image_shape[-2])
        z = torch.arange(image_shape[-1])
        
        # Randomly sample indices for each dimension
        sample_size_x = int(image_shape[-3] * sample_fraction)
        sample_size_y = int(image_shape[-2] * sample_fraction)
        sample_size_z = int(image_shape[-1] * sample_fraction)
        
        sampled_x = torch.sort(torch.randperm(len(x))[:sample_size_x])[0]
        sampled_y = torch.sort(torch.randperm(len(y))[:sample_size_y])[0]
        sampled_z = torch.sort(torch.randperm(len(z))[:sample_size_z])[0]

        # Create a meshgrid of coordinates for the sampled 3D image
        X, Y, Z = torch.meshgrid(sampled_x, sampled_y, sampled_z, indexing='ij')
        flat_X = X.flatten().float()
        flat_Y = Y.flatten().float()
        flat_Z = Z.flatten().float()

        # Compute pairwise differences in batches
        differences_x = flat_X[:, None] - flat_X[None, :]
        differences_y = flat_Y[:, None] - flat_Y[None, :]
        differences_z = flat_Z[:, None] - flat_Z[None, :]

        # Compute standard deviation
        sigma_x = torch.std(differences_x)
        sigma_y = torch.std(differences_y)
        sigma_z = torch.std(differences_z)

        return (sigma_x, sigma_y, sigma_z)
    
    def _compute_std_apperance(self, image, sample_fraction=0.5):
        """
        Computes std(x_i - x_j) using broadcasting in torch, with sampling a fraction of indexes from each dimension.
        Args:
        - image: Input 3D tensor (image).
        - sample_fraction: Fraction of indexes to sample from each dimension (default 0.5).
        """
        # Get the dimensions of the input image
        dims = image.shape[2:]

        # Calculate the number of indexes to sample per dimension
        sample_size = [int(dim * sample_fraction) for dim in dims]

        # Randomly sample indices from each dimension
        sampled_indices_x = sorted(random.sample(range(dims[0]), sample_size[0]))
        sampled_indices_y = sorted(random.sample(range(dims[1]), sample_size[1]))
        sampled_indices_z = sorted(random.sample(range(dims[2]), sample_size[2]))
        
        # Create the sampled image by indexing into the original image
        sampled_image = image[:, :, sampled_indices_x, :][:, :, :, sampled_indices_y, :][:, :, :, :, sampled_indices_z]
        
        # Flatten the sampled image
        flat_sampled_image = sampled_image.flatten()
        
        # Compute pairwise differences using broadcasting
        differences = flat_sampled_image[:, None] - flat_sampled_image[None, :]
        
        # Compute and return the standard deviation of the pairwise differences
        return torch.std(differences)

    def _initialize_sigmas(self, image):
        s_x, s_y, s_z = self.compute_std_spatial_sampled(image, sample_fraction=0.2) # 0.2 for real data
        s_a = self._compute_std_apperance(image, sample_fraction=0.2) # 0.2 for real data
        print('spatial, apperance std:', s_x, s_y, s_z, s_a)
    
        self.alpha = torch.tensor([s_x, s_y, s_z], dtype=torch.float32)*config.a_factor
        self.beta  = torch.tensor([s_a],          dtype=torch.float32)*config.b_factor
        self.gamma = torch.tensor([s_x, s_y, s_z], dtype=torch.float32)*config.g_factor
        self.spatial_ker_weight = config.spatial_ker_weight
        self.bilateral_ker_weight = config.bilateral_ker_weight
