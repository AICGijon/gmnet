import torch.nn.functional as F
import numpy as np
import torch


def JSD_Loss(p, p_hat):
    m = 0.5 * (p + p_hat)
    # compute the JSD Loss
    return 0.5 * (F.kl_div(p.log(), m) + F.kl_div(p_hat.log(), m))


class MRAE:
    def __init__(self, eps, n_classes):
        self.eps = eps
        self.n_classes = n_classes

    def __call__(self, p, p_hat):
        """MRAE implementation following Sebastiani et al. (2021) Evaluation Measures

        Args:
            p (torch.Tensor): True prevalences
            p_hat (torch.Tensor): Predicted prevalences
            eps ([type]): value for smothing. Suggested value in the paper is 1/2*n_samples

        Returns:
            [type]: [description]
        """
        if len(p.shape) != len(p_hat.shape):
            raise ValueError("The shape does not match")
        else:
            dims = len(p.shape)
        p_s = (p + self.eps) / (self.eps * self.n_classes + 1)
        p_hat_s = (p_hat + self.eps) / (self.eps * self.n_classes + 1)
        return (abs(p_s - p_hat_s) / p_s).mean(dims - 1).mean()


class MAE:
    def __init__(self, classes_to_monitor=None):
        self.classes_to_monitor = classes_to_monitor

    def __call__(self, p, p_hat):
        if self.classes_to_monitor is not None:
            return F.l1_loss(p[:, self.classes_to_monitor], p_hat[:, self.classes_to_monitor])
        else:
            return F.l1_loss(p, p_hat)


class MASE:
    def __init__(self, p_naive, classes_to_monitor="all"):
        self.p_naive = p_naive
        if classes_to_monitor == "all":
            self.classes_to_monitor = np.ones(len(p_naive))
        else:
            self.classes_to_monitor = classes_to_monitor

    def MASE(self, p, p_hat):
        mae_naive = (p[:, self.classes_to_monitor] - self.p_naive[self.classes_to_monitor]).abs().mean(axis=0)
        mae = (p[:, self.classes_to_monitor] - p_hat[:, self.classes_to_monitor]).abs().mean(axis=0)
        return (mae / mae_naive).mean()
    
class NMD:
    def __call__(self,prevs,prevs_hat):
        """
        Computes the Normalized Match Distance; which is the Normalized Distance multiplied by the factor
        `1/(n-1)` to guarantee the measure ranges between 0 (best prediction) and 1 (worst prediction).
        """
        P = torch.cumsum(prevs, dim=1)
        P_hat = torch.cumsum(prevs_hat, dim=1)
        distances = torch.abs(P - P_hat)
        match_distance = distances[:, :-1].sum(dim=1)
        n = prevs.shape[1]
        return ((1. / (n - 1)) * match_distance).mean()
