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

    def forward(self, x):
        mvn = []
        for i in range(self.num_gaussians):
            multivariate_normal = MultivariateNormal(self.centers[i], self.covariance[i])
            mvn.append(multivariate_normal)
        
        likelihoods = torch.stack([torch.exp(mn.log_prob(x)) for mn in mvn], dim=2)
    
        return likelihoods
