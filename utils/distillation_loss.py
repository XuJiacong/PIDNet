import torch
import torch.nn.functional as F
from torch import nn
from scipy.stats import norm
import math

class DistillationLoss:
    def __init__(self):
        # Dictionary to access functions by name
        self.loss_functions = {
            'cs': self.distill_loss_cosine_similarity,
            'norm_mse': self.distill_loss_mse_normalized,
        }

    def get_loss_function(self, func_name):
        """Retrieve a loss function by name."""
        return self.loss_functions.get(func_name, None)
    
    @staticmethod
    def distill_loss_cosine_similarity(s_feats, t_feats):
        kd_loss = 0
        for feat, t_feat in zip(s_feats, t_feats):
            feat = feat.view(feat.size(0), -1).double()
            t_feat = t_feat.view(t_feat.size(0), -1).double()
            kd_loss += torch.mean(1 - torch.nn.functional.cosine_similarity(feat, t_feat))
        return kd_loss

    @staticmethod
    def distill_loss_mse_normalized(s_feats, t_feats):
        kd_loss = 0
        for feat, t_feat in zip(s_feats, t_feats):
            # Normalize the features
            feat = (feat - feat.mean()) / (feat.std() + 1e-8)
            t_feat = (t_feat - t_feat.mean()) / (t_feat.std() + 1e-8)

            kd_loss += torch.nn.functional.mse_loss(feat, t_feat)
        return kd_loss