import torch
import numpy as np


def mae(y_true, y_pred):
    return torch.mean(torch.abs(y_true - y_pred))

def mse(y_true, y_pred):
    return torch.mean((y_true - y_pred) ** 2)

def rmse(y_true, y_pred):
    return torch.sqrt(mse(y_true, y_pred))

def r2_score(y_true, y_pred):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    mean = np.mean(y_true)
    ss_tot = np.sum((y_true - mean) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2

# $δ1$ accuracy
def delta1_accuracy(y_true, y_pred):
    y_true = y_true.cpu().numpy()+1
    y_pred = y_pred.cpu().numpy()+1
    delta = 1.25
    c = np.maximum(y_true / y_pred, y_pred / y_true)
    count = np.sum(c <= delta)
    return count / np.size(y_true)


if __name__ == '__main__':
    y_true = torch.tensor([[1, 2, 3, 4, 5],[1, 2, 3, 4, 5]], dtype=torch.float32)
    y_pred = torch.tensor([[1.1, 2.1, 3.1, 4.1, 5.1],[1.1, 2.1, 3.1, 4.1, 5.1]], dtype=torch.float32)
    print(mae(y_true, y_pred))
    print(mse(y_true, y_pred))
    print(rmse(y_true, y_pred))
    print(r2_score(y_true, y_pred))
    print(delta1_accuracy(y_true, y_pred))