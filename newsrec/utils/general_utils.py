# -*- coding: utf-8 -*-
# @Author        : Rui
# @Time          : 2024/4/8 21:34
# @Function      : General utils functions
import os
import json
import torch
import random
import copy
import importlib

from pathlib import Path
from collections import defaultdict
from typing import Union, Dict

from torch.nn.utils.rnn import pad_sequence


def read_json(file: Union[str, os.PathLike]):
    """
    Read json from file
    :param file: the path to the json file
    :return: ordered dictionary content
    """
    file = Path(file)
    with file.open("rt") as handle:
        return json.load(handle)


def write_json(content: Dict, file: Union[str, os.PathLike]):
    """
    Write content to a json file
    :param content: the content dictionary
    :param file: the path to save json file
    """
    file = Path(file)
    with file.open("wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def get_project_root(project_name="Transformers4NewsRec", **kwargs):
    """
    Get the project root path
    :param project_name: Transformers4NewsRec
    :return: path of the project root
    """
    project_root = kwargs.get("project_root")
    if project_root and Path(project_root).exists():
        return project_root
    file_parts = Path(os.getcwd()).parts
    try:
        index = file_parts.index(project_name)
    except ValueError:
        return os.getcwd()
    abs_path = Path(f"{os.sep}".join(file_parts[: index + 1]))
    return os.path.relpath(abs_path, os.getcwd())


def init_obj(module_name: str, module_config: dict, module_class: object, *args, **kwargs):
    """
    Finds a function handle with the name given as 'type' in config, and returns the
    instance initialized with corresponding arguments given.

    `object = init_obj('Baseline', module, a, b=1)`
    is equivalent to
    `object = module.module_name(a, b=1)`
    :param module_name: name of the module
    :param module_config: configuration
    :param module_class: module class
    :param args: extra arguments
    :param kwargs: extra configurations
    """
    module_args = copy.deepcopy(module_config)
    module_args.update(kwargs)  # update extra configuration
    return getattr(module_class, module_name)(*args, **module_args)


def init_model_class(model_name, model_config, *args, **kwargs):
    """
    Initialize model class and return the instance
    :param model_name: name of the model class, BaseNRS
    :param model_config: configuration of the model
    :param args: extra arguments
    :param kwargs: extra configurations
    :return:
    """
    # setup model class
    model_module = importlib.import_module("newsrec.model")
    return init_obj(model_name, model_config, model_module, *args, **kwargs)


def pad_feat(input_feat):
    input_pad = {}
    for k, v in input_feat.items():
        try:
            input_pad[k] = pad_sequence(v, batch_first=True)
        except (IndexError, RuntimeError):
            input_pad[k] = torch.stack(v)
    return input_pad


def collate_fn(data):
    input_feat = defaultdict(lambda: [])
    for feat in data:
        for k, v in feat.items():
            input_feat[k].append(v)
    return pad_feat(input_feat)


def news_sampling(news, ratio):
    """Sample ratio samples from news list.
    If length of news is less than ratio, pad zeros.

    Args:
        news (list): packed_input news list
        ratio (int): sample number

    Returns:
        list: output of sample list.
    """
    if ratio > len(news):
        return news + [0] * (ratio - len(news))
    else:
        return random.sample(news, ratio)


def reshape_tensor(input_tensor, output_shape=None):
    """

    :param input_tensor: input tensor to reshape
    :param output_shape: output_shape that the shape of input_tensor will be reshaped to
    :return: tensor with the shape of output_shape (default: [batch_size*input_tensor.shape[1], ...])
    """
    if output_shape is None:
        output_shape = torch.Size([-1]) + input_tensor.size()[2:]
    return input_tensor.contiguous().view(output_shape)


