# -*- coding: utf-8 -*-
# @Author        : Rui
# @Time          : 2024/4/15 11:38
# @Function      : Define the loss functions
import torch
import torch.nn as nn
import torch.nn.functional as F


def nll_loss(predict, target):
    return F.nll_loss(predict, target)


def cross_entropy(predict, target):
    return F.cross_entropy(predict, target)


def nce_loss(predict, target):
    if len(target.shape) == 2:
        target = target.argmax(dim=-1)
    return F.nll_loss(F.log_softmax(predict, dim=-1), target)


def categorical_loss(predict, target, epsilon=1e-12):
    """
    Computes cross entropy between target (encoded as one-hot vectors) and prediction.
    Input: prediction (N, n) ndarray
           target (N, n) ndarray
    Returns: scalar
    """
    predict, target = F.softmax(predict.float(), dim=-1), target.float()
    predict = torch.clamp(predict, epsilon, 1. - epsilon)
    return -torch.sum(target * torch.log(predict + 1e-9)) / predict.shape[0]


def kl_divergence(predict, target):
    return F.kl_div(predict, target)


def bce_loss(predict, target, smooth_lambda=10):
    predict, target = predict.float(), target.float()
    return nn.BCELoss()(nn.Softmax(dim=-1)(smooth_lambda * predict), target)
