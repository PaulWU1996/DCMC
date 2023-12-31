import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

def euclidean_distance(coord1, coord2):
    mse = nn.MSELoss()
    return mse(coord1, coord2) # torch.norm(coord1 - coord2)

def generate_costmat(prediction, groundtruth):
    cost_mat = torch.zeros((len(prediction), len(groundtruth)))
    for i in range(len(prediction)):
        for j in range(len(groundtruth)):
            distance = euclidean_distance(prediction[i], groundtruth[j])
            cost_mat[i, j] = distance
    return cost_mat

def eval(batch_prediction, batch_groundtruth, threshold=0.5):
    batch_size = batch_prediction.shape[0]
    sample_per_batch = batch_prediction.shape[1]
    correct = 0
    total = batch_size * sample_per_batch
    for i in range(batch_size):
        prediction = batch_prediction[i]
        groundtruth = batch_groundtruth[i]
        cost_mat = generate_costmat(prediction, groundtruth)
        row_ind, col_ind = linear_sum_assignment(cost_mat.cpu().detach().numpy())
        correct = correct + torch.sum(cost_mat[row_ind, col_ind]<threshold).item()

    return correct, total