from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn

try:
    import permuto_cpp
except ImportError as e:
    raise (e, "Did you import `torch` first?")

_CPU = torch.device("cpu")
_EPS = np.finfo("float").eps


class PermutoFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q_in, features):
        q_out = permuto_cpp.forward(q_in, features)[0]
        ctx.save_for_backward(features)
        return q_out

    @staticmethod
    def backward(ctx, grad_q_out):
        feature_saved = ctx.saved_tensors[0]
        grad_q_back = permuto_cpp.backward(grad_q_out.contiguous(), feature_saved.contiguous())[0]
        return grad_q_back, None 
    

def _spatial_features(image, v, sigma):
    """
    Return the spatial features as a Tensor

    Args:
        image:  Image as a Tensor of shape (channels, depth, height, width)
        v:      Scaling factors for spatial dimensions (z, y, x)
        sigma:  Bandwidth parameters for spatial dimensions (z, y, x)

    Returns:
        Tensor of shape [d, h, w, 3] with spatial features
    """
    _, d, h, w = image.size()
    z = torch.arange(start=0, end=d, dtype=torch.float32, device=image.device).view(-1, 1, 1)
    zz = v[0] * z.repeat([1, h, w]) / sigma[0]

    x = torch.arange(start=0, end=w, dtype=torch.float32, device=image.device)
    xx = v[1] * x.repeat([d, h, 1]) / sigma[1]

    y = torch.arange(start=0, end=h, dtype=torch.float32, device=image.device).view(1, -1, 1)
    yy = v[2] * y.repeat([d, 1, w]) / sigma[2]

    return torch.stack([zz, yy, xx], dim=3)  # Shape: [d, h, w, 3]

class AbstractFilter(ABC, nn.Module):
    """
    Super-class for permutohedral-based Gaussian filters
    """

    def __init__(self, image):
        super(AbstractFilter, self).__init__()  # Initialize nn.Module
        self.image = image

    def apply(self, input_):
        self.features = self._calc_features(self.image) # recalculate features since bandwidth changing?
        output = PermutoFunction.apply(input_, self.features)
        return output * self._calc_norm(self.image)

    @abstractmethod
    def _calc_features(self, image):
        pass

    def _calc_norm(self, image):
        _, d, h, w = image.size()
        all_ones = torch.ones((1, d, h, w), dtype=torch.float32, device=image.device)
        norm = PermutoFunction.apply(all_ones, self.features)
        return 1.0 / (norm + _EPS)


class SpatialFilter(AbstractFilter):
    """
    Gaussian filter in the spatial ([z, y, x]) domain
    """

    def __init__(self, image, sigma_gamma):
        """
        Create new instance

        Args:
            image:        Image tensor of shape (channels, depth, height, width)
            sigma_gamma:  Bandwidth parameters for spatial dimensions (z, y, x)
        """
        super(SpatialFilter, self).__init__(image)  # Initialize AbstractFilter
        self.sigma_gamma = sigma_gamma
        self.v_gamma = nn.Parameter(torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32))  # Shape: [3]

    def _calc_features(self, image):
        return _spatial_features(image, self.v_gamma, self.sigma_gamma)


class BilateralFilter(AbstractFilter, nn.Module):
    """
    Gaussian filter in the bilateral ([r, g, b, z, y, x]) domain
    """

    def __init__(self, image, sigma_alpha, sigma_beta):
        """
        Create new instance

        Args:
            image:        Image tensor of shape (channels, depth, height, width)
            sigma_alpha:  Bandwidth parameters for spatial dimensions (z, y, x)
            sigma_beta:   Bandwidth parameter for color dimensions (r, g, b)
        """
        super(BilateralFilter, self).__init__(image)  # Initialize AbstractFilter
        self.sigma_alpha = sigma_alpha
        self.sigma_beta = sigma_beta
        self.v_alpha = nn.Parameter(torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32))  # Shape: [3]
        self.v_beta = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))  # Shape: [1]

    def _calc_features(self, image):
        # Spatial features (z, y, x)
        xyz = _spatial_features(image, self.v_alpha, self.sigma_alpha)  # Shape: [d, h, w, 3]

        # Color features (r, g, b)
        rgb = (self.v_beta * image.permute(1, 2, 3, 0) / float(self.sigma_beta))  # Shape: [d, h, w, 1]

        # Concatenate spatial and color features
        return torch.cat([xyz, rgb], dim=3)  # Shape: [d, h, w, 4]