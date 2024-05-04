# -*- coding: utf-8 -*-
# @Author        : Rui
# @Time          : 2024/4/29 19:28
# @Function      :
import torch
import torch.nn as nn
from newsrec.data import load_news_graph, load_entity_graph
from newsrec.model.general import MultiHeadedAttention, AttLayer
from torch_geometric.nn import Sequential, GatedGraphConv
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
from .base import BaseNRS


class GLORYRSModel(BaseNRS):
    def __init__(self, **kwargs):
        self.head_num, self.head_dim = kwargs.get("head_num", 20), kwargs.get("head_dim", 20)
        self.embedding_dim = kwargs.get("embedding_dim", self.head_num * self.head_dim)
        super().__init__(**kwargs)
        word_embed_dim = self.word_embedding.embed_dim
        self.local_news_encoder = Sequential("x,mask", [
            (MultiHeadedAttention(self.head_num, self.head_dim, word_embed_dim), "x,x,x,mask -> x,x_att"),
            (nn.LayerNorm(self.embedding_dim), "x->x"),
            (self.dropout_ne, "x->x"),
            (AttLayer(self.embedding_dim, self.attention_hidden_dim), "x,mask -> x,x_att"),
            # (nn.LayerNorm(self.embedding_dim), "x,x_att->x,x_att"),
        ])
        self.news_graph = load_news_graph(**kwargs)
        self.num_neighbors = kwargs.get("num_neighbors", 8)
        neighbor_matrix = torch.zeros(len(self.news_graph.neighbor_mapper)+1, self.num_neighbors, dtype=torch.int32)
        for k, v in self.news_graph.neighbor_mapper.items():
            neighbor_matrix[k, :min(self.num_neighbors, len(v))] = torch.tensor(v[:self.num_neighbors])
        self.news_neighbors = nn.Embedding.from_pretrained(neighbor_matrix, freeze=True)
        self.k_hops = kwargs.get("k_hops", 2)
        if kwargs.get("entity_feature"):
            self.entity_graph = load_entity_graph(**kwargs)
        self.global_news_encoder = Sequential("x,index", [
            (GatedGraphConv(self.embedding_dim, num_layers=3, aggr="add"), "x,index -> x"),
        ])
        self.aggregator_encoder = AttLayer(self.embedding_dim, self.attention_hidden_dim)
        self.user_layer = Sequential("x,mask", [
            (MultiHeadedAttention(self.head_num, self.head_dim, self.embedding_dim), 'x,x,x,mask -> x,x_att'),
            (AttLayer(self.embedding_dim, self.attention_hidden_dim), "x,mask -> x,x_att"),
        ])
        self.news_layer = None  # news_layer is not used, so set to None

    def build_input_feat(self, input_feat):
        history_nid, history_mask = input_feat["history_nid"], input_feat["history_mask"]
        candidate_nid, candidate_mask = input_feat["candidate_nid"], input_feat["candidate_mask"]
        candidate_selected = torch.masked_select(candidate_nid, candidate_mask == 1)  # select candidate based on mask
        current = torch.masked_select(history_nid, history_mask == 1)  # select history based on mask
        if current.size(0) == 0:
            current = torch.tensor([0], device=history_nid.device, dtype=torch.int32)
        history_neighbor_all = current
        for _ in range(self.k_hops):  # get the neighbors of the history news and append to all list
            current = self.news_neighbors(current)  # get current history news neighbors
            history_neighbor_all = torch.cat([history_neighbor_all, torch.masked_select(current, current != 0)])
        history_neighbor_all = torch.unique(history_neighbor_all)  # remove duplicates
        # subgraph = self.news_graph.build_subgraph(history_neighbor_all)
        edge_index = self.news_graph.graph_data.edge_index.to(history_nid.device)
        edge_attr = self.news_graph.graph_data.edge_attr.to(history_nid.device)
        sub_edge_index, sub_edge_attr = subgraph(history_neighbor_all, edge_index, edge_attr, relabel_nodes=True,
                                                 num_nodes=self.news_graph.graph_data.num_nodes)
        input_feat["sub_graph"] = Data(x=history_neighbor_all, edge_index=sub_edge_index, edge_attr=sub_edge_attr)
        input_feat["nid"] = torch.unique(torch.cat((history_neighbor_all, candidate_selected), dim=0))
        # -1 means zero padding; history_mapping: B, H; candidate_mapping: B, C; sub_graph_news_mapping: X
        input_feat["history_mapping"], input_feat["candidate_mapping"], input_feat["sub_graph_news_mapping"] = (
            self.get_mapping_index(input_feat["nid"], history_nid, candidate_nid, history_neighbor_all)
        )
        input_feat["history_graph_mapping"], = (
            self.get_mapping_index(input_feat["sub_graph_news_mapping"], input_feat["history_mapping"])
        )
        return input_feat

    def news_encoder(self, input_feat):
        word_vector, news_mask = self.text_feature_encoder(input_feat)  # shape = (B*(H+C), F, E)
        output = self.local_news_encoder(word_vector, news_mask)
        return {"news_vector": output[0], "news_weight": output[1]}

    def user_encoder(self, input_feat):
        local_history_news = input_feat["history_news"]
        sub_news_vector = self.get_mapping_vector(input_feat["news_vector"], input_feat["sub_graph_news_mapping"])
        graph_vector = self.global_news_encoder(sub_news_vector, input_feat["sub_graph"].edge_index)
        # (history_mapping, ) = self.get_mapping_index(input_feat["sub_graph_news_mapping"], input_feat["history_nid"])
        graph_vector_batch = self.get_mapping_vector(graph_vector, input_feat["history_graph_mapping"])
        user_vector = self.aggregator_encoder(torch.stack([local_history_news, graph_vector_batch], dim=-2))[0]
        user_vector, user_weight = self.user_layer(user_vector, input_feat["history_mask"])
        return {"user_vector": user_vector, "user_weight": user_weight}
