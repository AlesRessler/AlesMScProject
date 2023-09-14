import torch
import torch.nn as nn

from sphericalharmonics.utils import get_number_of_coefficients


def generate_mask(max_degree):
    number_of_coefficients = get_number_of_coefficients(max_degree)
    mask = torch.zeros(number_of_coefficients)

    start_index = 0

    for degree in range(0, max_degree + 2, 2):
        value = 2 * degree + 1
        end_index = start_index + value
        mask[start_index:end_index] = value
        start_index = end_index

    return mask


class CrossCorrelationLoss(nn.Module):
    def __init__(self, max_degree, device='mps'):
        super(CrossCorrelationLoss, self).__init__()

    def forward(self, predictions, targets):
        cross_correlation_loss = (torch.sum(torch.square(predictions), dim=1) + torch.sum(torch.square(targets), dim=1) - 2 * torch.sum(predictions * targets, dim=1))
        return cross_correlation_loss
