import numpy as np


def binary_classification_metrics(y_pred, y_true):
    """
    Computes metrics for binary classification
    Arguments:
    y_pred, np array (num_samples) - model predictions
    y_true, np array (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    """

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score

    precision = 0
    recall = 0
    f1 = 0

    tp = 0
    fp = 0
    fn = 0
    tn = 0

    for i in range(y_pred.size):
        if y_pred[i] == 1:
            if y_pred[i] == y_true[i]:
                tp += 1
            else:
                fp += 1
        else:
            if y_pred[i] == y_true[i]:
                tn += 1
            else:
                fn += 1

    if tp + fp != 0:
        precision = tp / (tp + fp)
    if tp + fn != 0:
        recall = tp / (tp + fn)
    if 2 * tp + fp + fn != 0:
        f1 = 2 * tp / (2 * tp + fp + fn)

    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return [precision, recall, f1, accuracy]


def multiclass_accuracy(y_pred, y_true):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    accuracy = (y_pred == y_true).sum() / len(y_pred)
    return accuracy


def r_squared(y_pred, y_true):
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """

    y = np.mean(y_true)
    r2 = 1 - np.power(y_pred - y_true, 2).sum() / np.power(y_true - y, 2).sum()
    return r2


def mse(y_pred, y_true):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """

    n = y_pred.size
    mse = np.power(y_pred - y_true, 2).sum() / n
    return mse


def mae(y_pred, y_true):
    """
    Computes mean absolut error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mae - mean absolut error
    """

    n = y_pred.size
    mae = np.absolute(y_pred - y_true).sum() / n
    return mae
    