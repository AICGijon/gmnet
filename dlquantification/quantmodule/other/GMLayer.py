import torch.nn as nn
import torch
from torch.distributions.multivariate_normal import MultivariateNormal


class GMLayer(nn.Module):
    def __init__(self, n_features, num_gaussians, device, requires_grad=True):
        super(GMLayer, self).__init__()
        self.n_features = n_features
        self.num_gaussians = num_gaussians
        self.device = device
        self.centers = nn.Parameter(
            torch.rand(num_gaussians, n_features), requires_grad=requires_grad
        )
        covariance = torch.eye(n_features).repeat(num_gaussians, 1, 1)
        self.covariance = nn.Parameter(covariance, requires_grad=requires_grad)

    def compute_likelihoods(self, x):
        # x: (100, 1000, 5)
        # centers: (num_gaussians, 5)
        # covariance: (num_gaussians, 5, 5)

        # Get the number of Gaussians and features
        feature_dim = self.centers.shape[1]

        # Expand centers and covariance to match x's batch size and example count
        # centers_expanded: (1, 1, num_gaussians, feature_dim)
        # x_expanded: (100, 1000, 1, feature_dim)
        centers_expanded = self.centers.unsqueeze(0).unsqueeze(0)
        x_expanded = x.unsqueeze(2)

        # Difference: (100, 1000, num_gaussians, feature_dim)
        diff = x_expanded - centers_expanded

        # Covariance inverse and determinant
        # cov_inv: (num_gaussians, feature_dim, feature_dim)
        # det_cov: (num_gaussians,)
        cov_inv = torch.inverse(self.covariance)
        det_cov = torch.linalg.det(self.covariance)

        # Mahalanobis term
        # diff: (100, 1000, num_gaussians, feature_dim)
        # cov_inv @ diff.unsqueeze(-1): (100, 1000, num_gaussians, feature_dim, 1)
        # mahalanobis: (100, 1000, num_gaussians)
        mahalanobis = torch.einsum(
            '...i,...ij,...j->...',
            diff,
            cov_inv.unsqueeze(0).unsqueeze(0),
            diff
        )

        # Gaussian log-probability formula
        # likelihoods: (100, 1000, num_gaussians)
        normalization_term = (torch.log((2 * torch.pi) ** feature_dim * det_cov)).unsqueeze(0).unsqueeze(0)
        log_probs = -0.5 * (mahalanobis + normalization_term)

        return torch.exp(log_probs)

    def forward(self, x):
        # mvn = []
        # for i in range(self.num_gaussians):
        #     multivariate_normal = MultivariateNormal(self.centers[i], self.covariance[i])
        #     mvn.append(multivariate_normal)
        
        # likelihoods = torch.stack([torch.exp(mn.log_prob(x)) for mn in mvn], dim=2)
    
        # return likelihoods
        return self.compute_likelihoods(x)
