import numpy as np
from sklearn import metrics

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def std(y):
    return np.sqrt( np.sum( (y - y.mean()) * (y - y.mean())) / len(y) )

def CORR(pred, true):
    # u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    # d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    # return (u / d).mean(-1)
    A = ((true - true.mean()) * (pred - pred.mean())).mean()
    B = std(true) * std(pred)
    corr = A / B
    return corr




def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


# def MAPE(pred, true):
#     return np.mean(np.abs((pred - true) / true))

# def MAPE(pred,true):
#     return metrics.mean_absolute_percentage_error(true, pred)

def MAPE(y_pred, y_true, null_val=0):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(y_pred, y_true).astype('float32'),y_true))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100



def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)
    corr = CORR(pred, true)

    return mae, mse, rmse, mape, mspe, rse, corr