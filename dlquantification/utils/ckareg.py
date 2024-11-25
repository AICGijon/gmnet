import torch
import numpy as np


class CKARegularization:
    def feature_space_linear_cka(self, latent_spaces):
        cka = 0
        counter = 0
        for i in range(len(latent_spaces)):
            for j in range(i + 1, len(latent_spaces)):
                features_x = latent_spaces[i]
                features_y = latent_spaces[j]
                features_x = features_x - torch.mean(features_x, dim=0, keepdim=True)
                features_x = features_x - torch.mean(features_x, dim=0, keepdim=True)
                features_y = features_y - torch.mean(features_y, dim=0, keepdim=True)

                dot_product_similarity = torch.norm(torch.matmul(features_x.t(), features_y)) ** 2
                normalization_x = torch.norm(torch.matmul(features_x.t(), features_x))
                normalization_y = torch.norm(torch.matmul(features_y.t(), features_y))
                cka += dot_product_similarity / (normalization_x * normalization_y)
                counter += 1
        return cka / counter
