import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as P

import scipy


def time_weighted_power_loss(y, y_hat, loss=torch.nn.MSELoss(), p=2):
    """_summary_

    Args:
        y and y_hat of shape T * B
        y (_type_): true label
        y_hat (_type_): prediction
        loss (_type_, optional): loss function. Defaults to torch.nn.MSELoss().
        p (int, optional): supported [-1, 0, 1, 2, np.inf]. Defaults to 2.

    Returns:
        _type_: time weighted loss value,
                if p == 0, return the average value of original loss value
    """
    if not np.isinf(p):
        weighted_loss = 0
        length = y.shape[-2]

        scale = sum((i + 1) ** p for i in range(length))

        for i in range(length):
            weighted_loss += loss(y[i, :], y_hat[i, :]) * (i + 1) ** p / scale

        return weighted_loss
    else:
        weighted_loss = loss(y[-1, :], y_hat[-1, :])
        return weighted_loss


class ExpLambda(nn.Module):
    def forward(self, X):
        return torch.matrix_exp(X)  # Return a exponential matrix

    def right_inverse(self, A):
        return torch.real(torch.from_numpy(scipy.linalg.logm(A)))




