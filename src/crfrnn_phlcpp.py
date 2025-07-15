import torch
import torch.nn as nn

from src.phlcpp_filters import SpatialFilter, BilateralFilter
from src.phlcpp_params import DenseCRFParams

class CrfRnn3D_phlcpp(nn.Module):
    """
    PyTorch implementation of the CRF-RNN module described in the paper:

    Conditional Random Fields as Recurrent Neural Networks,
    S. Zheng, S. Jayasumana, B. Romera-Paredes, V. Vineet, Z. Su, D. Du, C. Huang and P. Torr,
    ICCV 2015 (https://arxiv.org/abs/1502.03240).
    """
    def __init__(self, num_labels, num_iterations=5, image=None, crf_init_params=None):
        """
        Create a new instance of the CRF-RNN layer.

        Args:
            num_labels:         Number of semantic labels in the dataset
            num_iterations:     Number of mean-field iterations to perform
            crf_init_params:    CRF initialization parameters
            image:              Beta of shape (_,_,d,h,w) 
        """
        super(CrfRnn3D_phlcpp, self).__init__()

        if crf_init_params is None:
            crf_init_params = DenseCRFParams(image)

        self.params = crf_init_params
        self.num_iterations = num_iterations
        self._softmax = torch.nn.Softmax(dim=0)
        self.num_labels = num_labels

        # --------------------------------------------------------------------------------------------
        # --------------------------------- Trainable Parameters -------------------------------------
        # --------------------------------------------------------------------------------------------

        # Spatial kernel weights
        self.spatial_ker_weights = nn.Parameter(
            crf_init_params.spatial_ker_weight
            * torch.eye(num_labels, dtype=torch.float32)
        )

        # Bilateral kernel weights
        self.bilateral_ker_weights = nn.Parameter(
            crf_init_params.bilateral_ker_weight
            * torch.eye(num_labels, dtype=torch.float32)
        )

        # Compatibility transform matrix
        self.compatibility_matrix = nn.Parameter(
            torch.eye(num_labels, dtype=torch.float32)
        )

        # --------------------------------------------------------------------------------------------
        # --------------------------------- Non-Trainable Parameters -------------------------------------
        # --------------------------------------------------------------------------------------------
        image = image.squeeze(0)  # TODO: match dimensions instead of squeezing this
        self.spatial_filter = SpatialFilter(image, sigma_gamma=self.params.gamma)
        self.bilateral_filter = BilateralFilter(image, sigma_alpha=self.params.alpha, sigma_beta=self.params.beta)

    def forward(self, U, I=None):
        """
        Perform CRF inference for 3D data.

        Args:
            U:  Tensor of shape (1, num_classes, D, H, W) containing the unary logits
            I:  image, unused here
        Returns:
            log-Q distributions (logits) after CRF inference
        """
        if U.shape[0] != 1:
            raise ValueError("Only batch size 1 is currently supported!")

        U = U.squeeze(0)  # Remove the first dimension if it's of size 1
        _, d, h, w = U.shape

        for _ in range(self.num_iterations):

            Q = self._softmax(U)

            # Spatial filtering
            spatial_out = torch.mm(
                self.spatial_ker_weights,
                self.spatial_filter.apply(Q).view(self.num_labels, -1),
            )

            # Bilateral filtering
            bilateral_out = torch.mm(
                self.bilateral_ker_weights,
                self.bilateral_filter.apply(Q).view(self.num_labels, -1),
            )

            # Compatibility transform
            Q = spatial_out + bilateral_out 

            Q = torch.mm(self.compatibility_matrix, Q).view(
                self.num_labels, d, h, w
            )

            # Adding unary potentials back
            Q = U + Q 

        Q = torch.unsqueeze(self._softmax(Q), 0)
        return Q
