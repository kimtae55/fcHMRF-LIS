import torch
import torch.nn as nn
from src.kde import GaussianKDE
from torch.distributions import Normal
import torch.nn.functional as F
from src.crfrnn_phlcpp import CrfRnn3D_phlcpp
import numpy as np
import matplotlib.pyplot as plt

torch.autograd.set_detect_anomaly(True) # Enable this for gradient debugging 

class fchmrf(nn.Module):
    """
    High-level module for fcHMRF-LIS. 
    It optimizes parameters through EM-algorithm and mean field inference. 

    Parameters:
    - im: test statistics 
    - beta: mean difference in voxel intensities
    """
    def __init__(self, im, beta, mask=None, pval=None, threshold=0.05, lr=1e-3): 
        super(fchmrf, self).__init__()
        self.I = im  # input test statistics
        self.I_shape = self.I.shape
        self.B = beta # mean difference in voxel intensities (beta_1 - beta_0 of a binary categorical variable for linear regression) 
        self.threshold = threshold
        self._softmax = torch.nn.Softmax(dim=1)

        # Initialize h using p-values
        self.p_value = 2.0 * (1.0 - torch.distributions.Normal(0, 1).cdf(im.abs()))  
        self.mask = None
        self.p_mask = 1.0 - self.p_value # fine for all simulations

        self.h = torch.rand((1,2,) + im.shape[-3:])
        self.h[:, 0:1, :, :, :] = torch.tensor(1.0-self.p_mask) # P(h=0|x)
        self.h[:, 1:2, :, :, :] = torch.tensor(self.p_mask)  # P(h=1|x)

        # Initialize CRF-RNN models for Q_1 and Q_2
        self.q_model = CrfRnn3D_phlcpp(num_labels=2, num_iterations=10, image=self.B) 

        # P(h=1|x) \approx Q(h=1|x) will be stored here
        self.q_hx_1 = None

        # Initialize additinoal model parameters
        self.w_0 = nn.Parameter(torch.tensor([0.0])) 
        self.N_01 = Normal(torch.tensor([0.0]), torch.tensor([1.0]))

        self.kde = GaussianKDE(self.I.flatten(), batch_size=1000)
        self.kde.update_weights(self.h[:,1:2,:,:,:].flatten(), update_kernel=True)
        self.f1_cont = self.kde.logpdf().reshape(self.I_shape)

        # Define the optimizer (shared across M-steps)
        self.optimizer = torch.optim.AdamW(self.parameters(),lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, 0.9)

    def bernoulli_sample(self, p):
        """
        Performs actual Bernoulli sampling (discrete 0 or 1) from given probabilities.

        Parameters
        ----------
        p : torch.Tensor
            Probability of sampling 1 (Bernoulli probability). Shape is [:, 1, :, :, :]

        Returns
        -------
        torch.Tensor
            Sampled value in {0, 1}. Shape is [:, 2, :, :, :]
        """
        p = p.squeeze(1)  # Keep dimensions consistent
        p = p.clamp(min=1e-6, max=1 - 1e-6)  # Avoid numerical issues
        # Bernoulli sampling (discrete 0 or 1)
        sample = torch.bernoulli(p)  # Shape [:, :, :, :]
        sample = torch.stack([1 - sample, sample], dim=1)  # Shape [:, 2, :, :, :]
        return sample

    def Q1(self):
        """
        Caclulates Q1 and stores optimal f1 to self.f1
        """
        U = -self.h*(self.w_0+self.N_01.log_prob(self.I)-self.f1_cont)
        # Saving unary potentials for next EM step 
        self.h = self.q_model(U, self.B)
        self.q_hx_1 = self.h[:, 1:2, :, :, :] # shape of q_hx_1 should be (B, N_filters, 1, D, H, W)
        self.q_hx_1 = self.q_hx_1.detach().clone()
        self.h = self.h.detach().clone()

        self.kde.update_weights(self.q_hx_1.flatten(), update_kernel=True)
        self.f1_cont = self.kde.logpdf().reshape(self.I_shape)
        return torch.sum(self.q_hx_1*self.f1_cont)

    def Q2(self, n_samples=1):
        """
        Calculates Q2 that is used for optimizing w
        n_samples = 100 is used for simulations in the paper, n_samples=1 also works well in practice
        """
        all_log_q = []

        for _ in range(n_samples):  # Sampling N times
            h_sample = self.bernoulli_sample(self.q_hx_1)  # Sample h
            U = -h_sample * self.w_0  # Transform with weights
            q_output = self.q_model(U, self.B)  # Compute q(h)
            # dim=1 means i'm selecting value of q_output based on class 0 or class 1
            # index of the dim (0 or 1) is chosen based on h_sample, which achieves q(h_i=h_(s))
            # gathered_q.shape = (1, 1, 30, 30, 30)
            indexes = h_sample[:,1:2].long() # taking the class 1 samples from h_sample (0 or 1s)
            gathered_q = q_output.gather(dim=1, index=indexes) 
            log_q = torch.log(gathered_q + 1e-10)

            all_log_q.append(log_q)

        # Stack, mean over samples, and sum over i
        stacked_log_q = torch.stack(all_log_q, dim=1)
        mean_log_q = torch.mean(stacked_log_q, dim=1)
        final_result = torch.sum(mean_log_q)
        return final_result
        
    def e_step(self):
        """
        Perform the E-step, calculating Q(phi | phi^t).
        """
        return self.Q1(), self.Q2(n_samples=1)
    
    def m_step(self, inner_iter=5):
        """
        Perform the M-step: Optimize model parameters based on Q with multiple gradient steps.
        
        Parameters:
        - q: Sufficient statistics from E-step
        - inner_iter: Number of gradient update steps
        """
        for iter in range(inner_iter):
            q = self.Q2(n_samples=1)  
            if self.mask is not None:
                masked_q = q * self.mask  
                loss = -masked_q.sum() / self.mask.sum()
            else:
                loss = -q / self.I.numel()
            self.optimizer.zero_grad()   # Reset gradients
            loss.backward()              # Compute gradients
            self.optimizer.step()        # Update model parameters
            self.scheduler.step()        # Update LR scheduler

        return loss 
                
    def em_step(self):
        """
        Run the EM algorithm.
        
        Parameters:
        - phi_init: Initial guess for phi
        - max_iter: Maximum number of iterations
        """
        Q1_t0, Q2_t0 = self.e_step()  # E-step, returns the initial Q1 and Q2
        self.m_step(inner_iter=1)      # M-step, maximizes Q2, Q1_t0 is already in closed form 
        return self.q_hx_1, -Q1_t0/self.I.numel(), -Q2_t0/self.I.numel()
    
    