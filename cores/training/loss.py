import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

class CustomLoss(nn.Module):
    def __init__(self, weight=1.0):
        super(CustomLoss, self).__init__()
        self.weight = weight
        self.mse = nn.MSELoss()

    def _euclidean_distance(self, coord1, coord2):
        self.mse = nn.MSELoss()
        return self.mse(coord1, coord2) #torch.norm(coord1 - coord2)

    def _generate_costmat(self, prediction, groundtruth):
        cost_mat = torch.zeros((len(prediction), len(groundtruth)))
        for i in range(len(prediction)):
            for j in range(len(groundtruth)):
                distance = self._euclidean_distance(prediction[i], groundtruth[j])
                cost_mat[i, j] = distance
        return cost_mat

    def forward(self, batch_prediction, batch_groundtruth):
        batch_size = batch_prediction.shape[0]
        loss = torch.zeros(1)
        for i in range(batch_size):
            prediction = batch_prediction[i]
            groundtruth = batch_groundtruth[i]
            cost_mat = self._generate_costmat(prediction, groundtruth)
            row_ind, col_ind = linear_sum_assignment(cost_mat.cpu().detach().numpy())
            min_cost = cost_mat[row_ind, col_ind].mean()
            loss = loss + min_cost
        loss = (loss / batch_size) * self.weight
        return loss