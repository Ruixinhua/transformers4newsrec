# -*- coding: utf-8 -*-
# @Author        : Rui
# @Time          : 2024/4/25 22:03
# @Function      :
import torch
import pickle
import itertools
import numpy as np

from pathlib import Path
from collections import Counter, defaultdict
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, subgraph

from newsrec.utils import load_dataset_from_csv, load_feature_mapper, get_project_root


def load_neighbor_data(graph_data, directed=True):
    """
    Load the neighbor mapper and weights from the graph data
    :param graph_data: the graph data, news_graph or entity_graph
    :param directed: whether the graph is directed
    :return: neighbor_mapper, neighbor_weights
    """
    neighbor_mapper = defaultdict(list)
    neighbor_weights = defaultdict(list)
    edge_index = graph_data.edge_index
    edge_attr = graph_data.edge_attr
    if directed is False:
        edge_index, edge_attr = to_undirected(edge_index, edge_attr)
    for i in range(1, graph_data.num_nodes):
        dst_edges = torch.where(edge_index[1] == i)[0]  # i as dst
        weights = edge_attr[dst_edges]
        neighbor_nodes = edge_index[0][dst_edges]  # neighbors as src
        sorted_weights, indices = torch.sort(weights, descending=True)
        neighbor_mapper[i] = neighbor_nodes[indices].tolist()
        neighbor_weights[i] = sorted_weights.tolist()
    return neighbor_mapper, neighbor_weights


class NewsGraph:

    def build_subgraph(self, subset):
        sub_edge_index, sub_edge_attr = subgraph(subset, self.graph_data.edge_index, self.graph_data.edge_attr,
                                                 relabel_nodes=True)
        sub_news_graph = Data(x=subset, edge_index=sub_edge_index, edge_attr=sub_edge_attr)
        return sub_news_graph

    def __init__(self, **kwargs):
        news_data = load_dataset_from_csv(f"news_{kwargs.get('subset_name')}")
        news_dict = dict(zip(news_data["news_id"], news_data["nid"]))
        user_interaction_data = load_dataset_from_csv(f"user_interaction_{kwargs.get('subset_name')}")
        user_history = dict(zip(user_interaction_data["uid"], user_interaction_data["history"]))
        news_graph_source = kwargs.get("news_graph_source", "train_users")
        if news_graph_source == "all_users":
            edge_list = [[int(h) for h in user_history[user].split()] for user in user_history]
        else:
            train_data = load_dataset_from_csv(f"train_{kwargs.get('subset_name')}")
            train_users = set(train_data["uid"])
            edge_list = [[int(h) for h in user_history[user].split()] for user in train_users]
        node_feat = np.asarray([0] + list(news_dict.values()))  # add zero for the padding nid
        graph_type, construct_method = kwargs.get("graph_type", "trajectory"), kwargs.get("construct_method", "end")
        # graph tye should be "trajectory" or "co_occurrence"; construct method should be "end"-connection, "full", "no"
        short_edges = []
        for edge in edge_list:
            if graph_type == "trajectory":
                short_edges.extend([(edge[i], edge[i + 1]) for i in range(len(edge) - 1)])
                if construct_method == "end":
                    if len(edge) > 0:
                        short_edges.append((edge[-1], 0))
            elif graph_type == "co_occurrence":
                for i in range(len(edge) - 1):
                    for j in range(i + 1, len(edge)):
                        short_edges.append((edge[i], edge[j]))
                        short_edges.append((edge[j], edge[i]))
                if construct_method == "end":
                    if len(edge) > 0:
                        short_edges.append((edge[-1], 0))
            else:
                raise ValueError("Graph type should be 'trajectory' or 'co_occurrence'")
        edge_weights = Counter(short_edges)
        unique_edges = list(edge_weights.keys())

        edge_index = torch.tensor(list(zip(*unique_edges)), dtype=torch.int64)
        edge_attr = torch.tensor([edge_weights[edge] for edge in unique_edges], dtype=torch.int16)

        self.graph_data = Data(
            x=torch.from_numpy(node_feat),
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=len(node_feat)
        )


def load_news_graph(**kwargs):
    """
    Load the news graph from the cached file or create a new one
    :param kwargs: subset_name, saved_news_graph_path
    :return: a NewsGraph object
    """
    default_name = (f"glory_news_graph_{kwargs.get('subset_name')}_{kwargs.get('news_graph_source', 'train_users')}_"
                    f"{kwargs.get('graph_type', 'trajectory')}_{kwargs.get('construct_method', 'end')}")
    default_path = f"{get_project_root()}/cached/{default_name}.bin"
    saved_news_graph_path = Path(kwargs.get("saved_news_graph_path", default_path))
    if saved_news_graph_path.exists() and kwargs.get("use_cached_news_graph"):
        with open(saved_news_graph_path, "rb") as file:
            news_graph = pickle.load(file)
    else:
        news_graph = NewsGraph(**kwargs)
        with open(saved_news_graph_path, "wb") as file:
            pickle.dump(news_graph, file)
    news_graph.neighbor_mapper, news_graph.neighbor_weights = load_neighbor_data(
        news_graph.graph_data, kwargs.get("directed", True)
    )
    return news_graph


class EntityGraph:
    def __init__(self, **kwargs):
        news_graph = load_news_graph(**kwargs)
        feature_mapper = load_feature_mapper(**kwargs)
        entity_indices = feature_mapper.get_entity_matrix()
        entity_edge_index = []
        news_edge_src, news_edge_dest = news_graph.graph_data.edge_index
        edge_weights = news_graph.graph_data.edge_attr.long().tolist()
        for i in range(news_edge_src.shape[0]):
            src_entities = entity_indices[news_edge_src[i]]
            dest_entities = entity_indices[news_edge_dest[i]]
            src_entities_mask = src_entities > 0
            dest_entities_mask = dest_entities > 0
            src_entities = src_entities[src_entities_mask]
            dest_entities = dest_entities[dest_entities_mask]
            edges = list(itertools.product(src_entities, dest_entities)) * edge_weights[i]
            entity_edge_index.extend(edges)
        edge_weights = Counter(entity_edge_index)
        unique_edges = list(edge_weights.keys())

        edge_index = torch.tensor(list(zip(*unique_edges)), dtype=torch.int64)
        edge_attr = torch.tensor([edge_weights[edge] for edge in unique_edges], dtype=torch.int16)
        edge_index, edge_attr = to_undirected(edge_index, edge_attr)

        self.graph_data = Data(x=torch.arange(len(feature_mapper.entity_dict)),
                               edge_index=edge_index,
                               edge_attr=edge_attr,
                               num_nodes=len(feature_mapper.entity_dict))


def load_entity_graph(**kwargs):
    """
    Load the entity graph from the cached file or create a new one
    :param kwargs: subset_name, saved_entity_graph_path
    :return: a EntityGraph object
    """
    entity_feature = kwargs.get("entity_feature")
    entity_feature = [entity_feature] if isinstance(entity_feature, str) else entity_feature
    entity_name = "title_abstract" if "ab_entity" in entity_feature else "title"
    default_name = f"glory_entity_graph_{entity_name}_{kwargs.get('subset_name')}"
    default_path = f"{get_project_root()}/cached/{default_name}.bin"
    saved_entity_graph_path = Path(kwargs.get("saved_entity_graph_path", default_path))
    if saved_entity_graph_path.exists() and kwargs.get("use_cached_entity_graph"):
        with open(saved_entity_graph_path, "rb") as file:
            entity_graph = pickle.load(file)
    else:
        entity_graph = EntityGraph(**kwargs)
        with open(saved_entity_graph_path, "wb") as file:
            pickle.dump(entity_graph, file)
    entity_graph.neighbor_mapper, entity_graph.neighbor_weights = load_neighbor_data(
        entity_graph.graph_data, kwargs.get("directed", True)
    )
    return entity_graph


if __name__ == "__main__":
    news_graph_small = load_news_graph(subset_name="small", graph_type="trajectory")
    entity_graph_small = load_entity_graph(subset_name="small", entity_feature=["entity"], entity_len=5,
                                           title_len=30, abstract_len=0, body_len=70)
    news_graph_small.build_subgraph([1, 2])
