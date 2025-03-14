import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import scipy.stats as stats
import time 
import torch
import torch.nn as nn
import numpy as np

class GaussianKDE(nn.Module):
    def __init__(self, dataset, batch_size=1000):
        """
        Initialize the Gaussian KDE for 1D data with batched computation.

        Parameters
        ----------
        dataset : torch.Tensor
            The dataset to estimate from. Shape: (n_data_points,).
        batch_size : int, optional
            Number of points to process per batch for memory efficiency.
        """
        super().__init__()

        if dataset.ndim != 1:
            raise ValueError("Dataset must be 1D for this implementation.")
        
        self.dataset = dataset
        self.n = dataset.shape[0]
        self.batch_size = batch_size

        # Default equal weights (since all data is used)
        self.weights = torch.ones(self.n, dtype=dataset.dtype) / self.n

        # Compute effective sample size
        self.update_weights(self.weights, update_kernel=True)

    def silvermans_factor(self):
        """Compute Silverman's bandwidth factor for 1D data."""
        std_dev = self.dataset.std()
        iqr = torch.quantile(self.dataset, 0.75) - torch.quantile(self.dataset, 0.25)
        sigma_hat = torch.min(std_dev, iqr / 1.34)  # Robust estimate of spread
        return 0.9 * sigma_hat * self.neff ** (-1.0 / 5.0)  # Silverman's rule
    
    def scotts_factor(self):
        """Compute Scott's bandwidth factor for 1D data."""
        return self.neff ** (-1.0 / (1 + 4))  # 1D KDE

    def update_weights(self, weights, update_kernel=True):
        """
        Update weights, recompute effective sample size, and optionally update the kernel matrix.

        Parameters
        ----------
        weights : torch.Tensor
            Weights for each data point. Shape: (n_data_points,).
        update_kernel : bool, optional
            Whether to recompute the kernel matrix (default: True).
        """
        if weights.shape[0] != self.n:
            raise ValueError("Weights must have the same length as the dataset.")

        self.weights = weights / weights.sum()  # Normalize
        self.neff = (self.weights.sum() ** 2) / (self.weights ** 2).sum()
        self.bandwidth = self.silvermans_factor()

        if update_kernel:
            self._compute_kernel_matrix()

    def _compute_kernel_matrix(self):
        """
        Compute the kernel matrix in batches, ensuring it is recomputed only when weights are updated.
        """
        self.kernel_matrix = torch.zeros((self.n, self.n), dtype=self.dataset.dtype)

        for i in range(0, self.n, self.batch_size):
            batch_end = min(i + self.batch_size, self.n)
            batch_points = self.dataset[i:batch_end]  # Shape: (batch_size,)

            diff = batch_points.unsqueeze(1) - self.dataset.unsqueeze(0)  # Shape: (batch_size, n_data)
            kernel_vals = torch.exp(-0.5 * (diff / self.bandwidth) ** 2) / (self.bandwidth * torch.sqrt(torch.tensor(2 * torch.pi)))
            
            self.kernel_matrix[i:batch_end] = kernel_vals

    def pdf(self, batch_size=None):
        """
        Evaluate the PDF at the dataset points using the updated kernel matrix in batches.

        Parameters
        ----------
        batch_size : int, optional
            Number of points to process per batch. If None, use the default batch size.

        Returns
        -------
        torch.Tensor
            The estimated PDF values at the dataset points. Shape: (n_data_points,).
        """
        if batch_size is None:
            batch_size = self.batch_size

        pdf_vals = torch.zeros(self.n, dtype=self.dataset.dtype)

        for i in range(0, self.n, batch_size):
            batch_end = min(i + batch_size, self.n)
            pdf_vals[i:batch_end] = torch.matmul(self.kernel_matrix[i:batch_end], self.weights)

        return pdf_vals

    def logpdf(self, batch_size=None):
        """
        Compute the log-PDF at the dataset points using the updated kernel matrix in batches.

        Parameters
        ----------
        batch_size : int, optional
            Number of points to process per batch. If None, use the default batch size.

        Returns
        -------
        torch.Tensor
            The log-PDF values at the dataset points. Shape: (n_data_points,).
        """
        if batch_size is None:
            batch_size = self.batch_size

        return torch.log(self.pdf(batch_size) + 1e-10)
