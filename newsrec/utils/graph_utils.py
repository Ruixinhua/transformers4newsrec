# -*- coding: utf-8 -*-
# @Author        : Rui
# @Time          : 2024/4/24 15:49
# @Function      :
import torch
import numpy as np

from pathlib import Path
from .general_utils import get_project_root


def kg_default_root():
    kg_root_path = Path(get_project_root()) / "dataset/data/kg/wikidata-graph"
    if not kg_root_path.exists():
        raise FileExistsError("Default directory is not found, please specify kg root")
    return kg_root_path


def load_embeddings_from_text(file):
    """load embeddings from file"""
    return [np.array([float(i) for i in line.strip().split('\t')]) for line in open(file, "r", encoding="utf-8")]


def construct_entity_embedding(**kwargs):
    kg_root_path = kwargs.get("kg_root_path", kg_default_root())
    zero_array = np.zeros(kwargs.get("entity_embedding_dim", 100))  # zero embedding
    entity_embedding_file = kwargs.get("entity_embedding", kg_root_path / "entity2vecd100.vec")
    relation_embedding_file = kwargs.get("relation_embedding", kg_root_path / "relation2vecd100.vec")
    entity_embedding = [zero_array] + load_embeddings_from_text(entity_embedding_file)
    relation_embedding = [zero_array] + load_embeddings_from_text(relation_embedding_file)
    return torch.FloatTensor(np.array(entity_embedding)), torch.FloatTensor(np.array(relation_embedding))