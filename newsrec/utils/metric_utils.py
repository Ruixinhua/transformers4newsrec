# -*- coding: utf-8 -*-
# @Author        : Rui
# @Time          : 2024/4/15 14:04
# @Function      : Define the metric functions for evaluation
import torch
import sys

import numpy as np
from transformers import EvalPrediction


def mean_mrr(prediction, label):
    """
    Compute the mean reciprocal rank (MRR) over multiple prediction/target pairs.
    The function supports single and multiple evaluations across numpy arrays and torch tensors.

    :param prediction: Predicted scores, either as a numpy array or a torch tensor.
    :type prediction: np.ndarray | torch.Tensor | list
    :param label: Ground-truth labels, either as a numpy array or a torch tensor.
    :type label: np.ndarray | torch.Tensor | list
    :return: Mean MRR score.
    :rtype: float

    This function handles both single evaluations and batches of prediction.
    If input is a list or a multidimensional array/tensor, it computes the MRR for each pair and then takes the average.
    """
    def mrr_score(predict, target):
        if isinstance(predict, (np.ndarray, list)) and isinstance(target, (np.ndarray, list)):
            order = np.argsort(predict)[::-1]
            rank_indices = np.arange(1, len(target) + 1)
        elif isinstance(predict, torch.Tensor) and isinstance(target, torch.Tensor):
            order = torch.argsort(predict, descending=True)
            rank_indices = torch.arange(1, len(target) + 1, device=target.device)
        else:
            raise TypeError("Input must be either a numpy.ndarray or a torch.Tensor")
        # Reorder target according to the sorted indices
        target = target[order]
        # Compute reciprocal rank scores
        rr_score = target / rank_indices
        # Compute and return the MRR score
        mrr = rr_score.sum() / target.sum()
        return mrr.item() if isinstance(mrr, torch.Tensor) else mrr
    if isinstance(label, list) or (isinstance(label, (np.ndarray, torch.Tensor)) and label.ndim > 1):
        scores = [mrr_score(p, l) for p, l in zip(prediction, label)]
        mean_score = torch.mean(torch.tensor(scores)) if isinstance(scores[0], torch.Tensor) else np.mean(scores)
        return round(mean_score.item(), 4)
    else:
        return round(mrr_score(prediction, label), 4)


def ndcg_score(prediction, label, n=10):
    """
    Compute the Normalized Discounted Cumulative Gain (NDCG) at rank K.

    :param prediction: Predicted scores, either as a numpy array or a torch tensor.
    :type prediction: np.ndarray | torch.Tensor
    :param label: Ground-truth labels, either as a numpy array or a torch tensor.
    :type label: np.ndarray | torch.Tensor
    :param n: Rank at which the NDCG should be calculated.
    :type n: int
    :return: The NDCG score computed at K.
    :rtype: float

    This function calculates the ideal DCG (IDCG) using the true labels sorted by themselves,
    representing the best possible DCG. Then, it calculates the actual DCG using the predicted scores.
    Finally, it normalizes the actual DCG by the IDCG to get the NDCG, providing a score between 0 and 1,
    where 1 signifies the perfect ranking.
    """
    def dcg_score(predict, target, k=10):
        k = min(len(target), k)
        if isinstance(predict, np.ndarray) and isinstance(target, np.ndarray):
            order = np.argsort(predict)[::-1]
            target = target[order][:k]
            gains = 2 ** target - 1
            discounts = np.log2(np.arange(1, len(target) + 1) + 1)
        elif isinstance(predict, torch.Tensor):
            order = torch.argsort(predict, descending=True)
            target = target[order][:k]
            gains = 2 ** target - 1
            discounts = torch.log2(torch.arange(1, len(target) + 1, device=target.device) + 1)
        else:
            raise TypeError("predict must be either a numpy.ndarray or a torch.Tensor")
        return (gains / discounts).sum()

    best = dcg_score(label, label, n)
    actual = dcg_score(prediction, label, n)
    return actual / best


def ndcg(prediction, label, k):
    """
    Compute the normalized discounted cumulative gain (NDCG) over multiple prediction/prediction pairs or a single pair.
    :param prediction: Predicted scores, either as numpy arrays, torch tensors, or lists thereof.
    :type prediction: np.ndarray | torch.Tensor | list
    :param label: Ground-truth labels, either as numpy arrays, torch tensors, or lists thereof.
    :type label: np.ndarray | torch.Tensor | list
    :param k: The number of evaluated items.
    :type k: int
    :return: NDCG score, averaged over all pairs if a list is provided.
    :rtype: float
    """
    if isinstance(label, list) or (isinstance(label, (np.ndarray, torch.Tensor)) and label.ndim > 1):
        scores = [ndcg_score(p, t, k) for p, t in zip(prediction, label)]
        mean_score = torch.mean(torch.tensor(scores)) if isinstance(scores[0], torch.Tensor) else np.mean(scores)
        return round(mean_score.item(), 4)
    else:
        return round(ndcg_score(prediction, label, k), 4)


def ndcg_5(prediction, label):
    return ndcg(prediction, label, 5)


def ndcg_10(prediction, label):
    return ndcg(prediction, label, 10)


def binary_clf_curve(prediction, label):
    """
    Calculate true and false positives per binary classification
    threshold (can be used for roc curve or precision/recall curve);
    the calculation makes the assumption that the positive case
    will always be labeled as 1
    Source: https://ethen8181.github.io/machine-learning/model_selection/auc/auc.html#Implementation
    Parameters
    ----------
    label : 1d ndarray, shape = [n_samples]
        True targets/labels of binary classification
    prediction : 1d ndarray, shape = [n_samples]
        Estimated probabilities or scores
    Returns
    -------
    tps : 1d ndarray
        True positives counts, index i records the number
        of positive samples that got assigned a
        score >= thresholds[i].
        The total number of positive samples is equal to
        tps[-1] (thus false negatives are given by tps[-1] - tps)
    fps : 1d ndarray
        False positives counts, index i records the number
        of negative samples that got assigned a
        score >= thresholds[i].
        The total number of negative samples is equal to
        fps[-1] (thus true negatives are given by fps[-1] - fps)
    thresholds : 1d ndarray
        Predicted score sorted in decreasing order
    References
    ----------
    GitHub: scikit-learn _binary_clf_curve
    - https://github.com/scikit-learn/scikit-learn/blob/ab93d65/sklearn/metrics/ranking.py#L263
    """

    # sort predicted scores in descending order
    # and also reorder corresponding truth values
    desc_score_indices = np.argsort(prediction)[::-1]
    prediction = prediction[desc_score_indices]
    label = label[desc_score_indices]

    # prediction typically consists of tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve
    distinct_indices = np.where(np.diff(prediction))[0]
    end = np.array([label.size - 1])
    threshold_indices = np.hstack((distinct_indices, end))

    thresholds = prediction[threshold_indices]
    tps = np.cumsum(label)[threshold_indices]

    # (1 + threshold_indices) = the number of positives
    # at each index, thus number of data points minus true
    # positives = false positives
    fps = (1 + threshold_indices) - tps
    return tps, fps, thresholds


def roc_auc_score(prediction, label):
    """
    Compute Area Under the Curve (AUC) from prediction scores
    Parameters
    ----------
    label : 1d ndarray, shape = [n_samples]
        True targets/labels of binary classification
    prediction : 1d ndarray, shape = [n_samples]
        Estimated probabilities or scores
    Returns
    -------
    auc : float
    """

    # ensure the target is binary
    if np.unique(label).size != 2:
        raise ValueError('Only two class should be present in label. ROC AUC score '
                         'is not defined in that case.')

    tps, fps, _ = binary_clf_curve(prediction, label)

    # convert count to rate
    tpr = tps / tps[-1]
    fpr = fps / fps[-1]

    # compute AUC using the trapezoidal rule;
    # appending an extra 0 is just to ensure the length matches
    zero = np.array([0])
    tpr_diff = np.hstack((np.diff(tpr), zero))
    fpr_diff = np.hstack((np.diff(fpr), zero))
    auc = np.dot(tpr, fpr_diff) + np.dot(tpr_diff, fpr_diff) / 2
    return auc


def group_auc(prediction, label):
    """
    Compute the area under the ROC curve
    :param label: np.ndarray | list
    :param prediction: np.ndarray | list
    :return: roc auc score
    """

    if isinstance(label, list) or (isinstance(label, np.ndarray) and label.ndim > 1):
        return round(np.mean([roc_auc_score(p, t) for p, t in zip(prediction, label)]).item(), 4)
    else:
        return round(roc_auc_score(prediction, label), 4)


def compute_metrics(eval_prediction: EvalPrediction, metric_list: list):
    """
    Compute metrics based on the provided metric functions.
    :param eval_prediction: EvalPrediction object containing prediction and labels.
    :param metric_list: List of metric functions to be computed.
    :return: Dictionary containing the computed metrics.
    """
    prediction = eval_prediction.predictions  # mask padding tokens
    label = eval_prediction.label_ids
    metrics = {
        metric: round(np.mean([getattr(sys.modules[__name__], metric)(p[p != -100], t[t != -100])
                               for p, t in zip(prediction, label)]), 4)
        for metric in metric_list
    }
    metrics["monitor_metric"] = round(np.mean(list(metrics.values())), 4)
    return metrics
