# -*- coding: utf-8 -*-
# @Author        : Rui
# @Time          : 2024/4/29 19:28
# @Function      : Implement the model of GLORY
import copy

import torch
import torch.nn as nn
from newsrec.data import load_news_graph, load_entity_graph
from newsrec.model.general import MultiHeadAttentionAdv, AttLayer, LateFusion, FusionAggregator
from torch_geometric.nn import Sequential, GatedGraphConv, GAT, GATConv, GraphSAGE
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
from .base import BaseNRS


class GLORYRSModel(BaseNRS):
    """
    ✨ Going Beyond Local: Global Graph-Enhanced Personalized News Recommendations
    Paper of GLORY: https://arxiv.org/pdf/2307.06576
    Source library: https://github.com/tinyrolls/GLORY
    """
    def __init__(self, **kwargs):
        self.head_num, self.head_dim = kwargs.get("head_num", 20), kwargs.get("head_dim", 20)
        self.embedding_dim = kwargs.get("embedding_dim", self.head_num * self.head_dim)
        super().__init__(**kwargs)
        word_embed_dim = self.word_embedding.embed_dim
        self.use_local_entity = kwargs.get("use_local_entity") is not None  # whether to use entity feature
        self.use_local_entity_only = kwargs.get("use_local_entity_only", False)
        self.use_local_news_only = kwargs.get("use_local_news_only", False)
        self.use_entity_graph = kwargs.get("use_entity_graph", False)
        self.use_entity_graph_only = kwargs.get("use_entity_graph_only", False)
        self.use_news_graph = kwargs.get("use_news_graph", False)
        self.use_news_graph_only = kwargs.get("use_news_graph_only", False)
        self.use_candidate_local_entity = kwargs.get("use_candidate_local_entity", False)
        self.use_candidate_entity_graph = kwargs.get("use_candidate_entity_graph", False)
        self.use_candidate_news_graph = kwargs.get("use_candidate_news_graph", False)
        self.use_fused_feature = kwargs.get("use_fused_feature", False)
        self.fusion_method = kwargs.get("fusion_method", None)  # weighted, max, avg
        self.gnn_model = kwargs.get("gnn_model", "GGNN")  # GGNN, GAT
        self.fusion_method = None if self.fusion_method == "None" else self.fusion_method
        single_feature_num = sum([
            self.use_local_entity_only, self.use_local_news_only, self.use_entity_graph_only, self.use_news_graph_only
        ])
        # check if there are more than two conditions of "only" flag is true
        if single_feature_num > 1:
            raise ValueError(
                f"Only one of the following flags can be set to True: "
                f"use_local_entity_only: {self.use_local_entity_only}; use_news_graph_only: {self.use_news_graph_only};"
                f"use_local_news_only: {self.use_local_news_only}; use_entity_graph_only: {self.use_entity_graph_only}")
        self.news_encode_layer = MultiHeadAttentionAdv(self.head_num, self.head_dim, word_embed_dim,
                                                       use_flash_att=self.use_flash_att)
        self.user_layer_name = kwargs.get("user_layer_name", "mha")
        # local news encoder
        if self.user_layer_name == "mha":
            self.user_encode_layer = MultiHeadAttentionAdv(self.head_num, self.head_dim, self.embedding_dim,
                                                           use_flash_att=self.use_flash_att)
        # global news graph encoder
        if self.use_news_graph or self.use_news_graph_only:
            self.news_graph = load_news_graph(**kwargs)
            news_neighbors_num = kwargs.get("news_neighbors_num", 8)
            self.news_neighbor_mapper = self.news_graph.neighbor_mapper
            neighbor_matrix = torch.zeros(len(self.news_neighbor_mapper) + 1, news_neighbors_num, dtype=torch.int32)
            for k, v in self.news_neighbor_mapper.items():
                neighbor_matrix[k, :min(news_neighbors_num, len(v))] = torch.tensor(v[:news_neighbors_num])
            self.news_neighbors = nn.Embedding.from_pretrained(neighbor_matrix, freeze=True)
            self.k_hops_ng = kwargs.get("k_hops_ng", 2)
            if self.gnn_model.lower() == "gat":
                gnn_model = GAT(self.embedding_dim, out_channels=self.embedding_dim, hidden_channels=64, num_layers=3)
            elif self.gnn_model.lower() == "gatconv":
                gnn_model = GATConv(self.embedding_dim, out_channels=self.embedding_dim)
            elif self.gnn_model.lower() == "graphsage":
                gnn_model = GraphSAGE(self.embedding_dim, 64, 3, self.embedding_dim)
            else:
                gnn_model = GatedGraphConv(self.embedding_dim, num_layers=3, aggr="add")
            self.global_news_encoder = Sequential("x,index", [
                (gnn_model, "x,index -> x"),
            ])
        self.use_entity_feature = ((self.entity_feature and len(self.entity_feature)) and
                                   (self.use_local_entity or self.use_local_entity_only))
        # local entity encoder
        if self.use_entity_feature:
            self.local_entity_encoder = Sequential("x, mask", [
                (nn.Dropout(p=kwargs.get("dropout_le", 0.2)), "x -> x"),
                (MultiHeadAttentionAdv(int(self.entity_dim / self.head_dim), self.head_dim, self.entity_dim,
                                       use_flash_att=self.use_flash_att),
                 "x,x,x,mask -> x,x_att"),
                (nn.Dropout(p=kwargs.get("dropout_le", 0.2)), "x -> x"),
                (AttLayer(self.entity_dim, self.attention_hidden_dim), "x,mask -> x,x_att"),
                (nn.Linear(self.entity_dim, self.embedding_dim), "x -> x"),
                nn.LeakyReLU(0.2),
            ])
        # global entity graph encoder
        if self.use_entity_graph or self.use_entity_graph_only:
            self.entity_graph = load_entity_graph(**kwargs)
            entity_neighbors_num = kwargs.get("entity_neighbors_num", 8)
            self.entity_neighbor_mapper = self.entity_graph.neighbor_mapper
            neighbor_matrix = torch.zeros(len(self.entity_neighbor_mapper)+1, entity_neighbors_num, dtype=torch.int32)
            for k, v in self.entity_neighbor_mapper.items():
                neighbor_matrix[k, :min(entity_neighbors_num, len(v))] = torch.tensor(v[:entity_neighbors_num])
            self.entity_neighbors = nn.Embedding.from_pretrained(neighbor_matrix, freeze=True)
            self.k_hops_eg = kwargs.get("k_hops_eg", 2)
            if self.gnn_model.lower() == "gat":
                gnn_model = GAT(self.entity_dim, out_channels=self.entity_dim, hidden_channels=64, num_layers=3)
            elif self.gnn_model.lower() == "gatconv":
                gnn_model = GATConv(self.entity_dim, out_channels=self.entity_dim)
            elif self.gnn_model.lower() == "graphsage":
                gnn_model = GraphSAGE(self.entity_dim, 64, 3, self.entity_dim)
            else:
                gnn_model = GatedGraphConv(self.entity_dim, num_layers=3, aggr="add")
            self.global_entity_encoder = Sequential("x,index", [
                (gnn_model, "x,index -> x"),
            ])
            self.global_entity_att = Sequential('x, mask', [
                (nn.Dropout(p=kwargs.get("dropout_eg", 0.2)), 'x -> x'),
                (MultiHeadAttentionAdv(self.head_num, self.head_dim, self.entity_dim,
                                       use_flash_att=self.use_flash_att), 'x,x,x,mask -> x,x_att'),
                (AttLayer(self.head_num * self.head_dim, self.attention_hidden_dim), "x,mask -> x,x_att"),
            ])
        # fusion method
        if self.fusion_method:
            self.late_fusion = LateFusion(self.fusion_method, self.head_num, self.head_dim)
            user_feature_num = sum([self.use_entity_feature, self.use_entity_graph, self.use_news_graph]) + 1
            self.fusion_aggregator = FusionAggregator(self.head_num, self.head_dim, user_feature_num)
        # fused attention layer
        if self.use_fused_feature:
            self.fused_attention = AttLayer(self.embedding_dim, self.attention_hidden_dim)

    def build_input_feat(self, input_feat):
        history_nid, history_mask = input_feat["history_nid"], input_feat["history_mask"]
        candidate_nid, candidate_mask = input_feat["candidate_nid"], input_feat["candidate_mask"]
        candidate_selected = torch.masked_select(candidate_nid, candidate_mask)  # select candidate based on mask
        current = torch.masked_select(history_nid, history_mask)  # select history based on mask
        if self.use_news_graph or self.use_news_graph_only:
            if current.size(0) == 0:
                current = torch.tensor([0], device=history_nid.device, dtype=torch.int32)
            history_neighbor_all = copy.deepcopy(current)
            for _ in range(self.k_hops_ng):  # get the neighbors of the history news and append to all list
                current = self.news_neighbors(current)  # get current history news neighbors
                history_neighbor_all = torch.cat([history_neighbor_all, torch.masked_select(current, current != 0)])
            history_neighbor_all = torch.unique(history_neighbor_all)  # remove duplicates
            input_feat["nid"] = torch.unique(torch.cat((history_neighbor_all, candidate_selected), dim=0))
            # -1 means zero padding; history_mapping: B, H; candidate_mapping: B, C; sub_graph_news_mapping: X
            input_feat["history_mapping"], input_feat["candidate_mapping"], input_feat["sub_graph_news_mapping"] = (
                self.get_mapping_index(input_feat["nid"], history_nid, candidate_nid, history_neighbor_all)
            )
            edge_index = self.news_graph.graph_data.edge_index.to(history_nid.device)
            edge_attr = self.news_graph.graph_data.edge_attr.to(history_nid.device)
            sub_edge_index, sub_edge_attr = subgraph(input_feat["nid"], edge_index, edge_attr, relabel_nodes=True,
                                                     num_nodes=self.news_graph.graph_data.num_nodes)
            input_feat["sub_graph"] = Data(x=input_feat["nid"], edge_index=sub_edge_index, edge_attr=sub_edge_attr)
        else:
            input_feat["nid"] = torch.unique(torch.cat((current, candidate_selected), dim=0))
            input_feat["history_mapping"], input_feat["candidate_mapping"] = self.get_mapping_index(
                input_feat["nid"], history_nid, candidate_nid
            )  # -1 means zero padding; history_mapping: B, H; candidate_mapping: B, C
        return input_feat

    def text_feature_encoder(self, input_feat):
        """
        Encode text feature of news using word embedding layer
        :param input_feat: nid = history_nid + candidate_nid (training); shape = (B*(H+C));
        :return: word vector of news, shape = (B*(H+C), L, E), L is text length; E is word embedding dimension
        """
        # concat history news and candidate news
        news_all_feature = self.feature_embedding(input_feat["nid"])  # shape = (B*(H+C), F)
        news_feature_dict = self.feature_embedding.select_feature(news_all_feature)
        news_tokens = torch.cat([news_feature_dict[feature] for feature in self.text_feature], dim=-1)
        # news_tokens: shape = (B*(H+C), F), F is the sum of feature used in news
        news_mask = torch.where(news_tokens == self.pad_token_id, 0, 1).to(news_tokens.device)  # shape = (B*(H+C), F)
        word_vector = self.dropout_we(self.word_embedding(news_tokens, news_mask))  # shape = (B*(H+C), F, E)
        output_dict = {"word_vector": word_vector, "news_mask": news_mask}
        if self.use_entity_feature:
            output_dict.update({"entity_vector": self.entity_embedding(news_feature_dict["entity"])})
        if self.use_entity_graph or self.use_entity_graph_only:
            entity = torch.masked_select(news_feature_dict["entity"], news_feature_dict["entity"] != 0)
            entity_neighbors_all = copy.deepcopy(entity)
            for _ in range(self.k_hops_eg):
                entity = self.entity_neighbors(entity)
                entity_neighbors_all = torch.cat([entity_neighbors_all, torch.masked_select(entity, entity != 0)])
            entity_neighbors_all = torch.unique(entity_neighbors_all)
            entity_mapping, = self.get_mapping_index(entity_neighbors_all, news_feature_dict["entity"])
            edge_index = self.entity_graph.graph_data.edge_index.to(entity.device)
            edge_attr = self.entity_graph.graph_data.edge_attr.to(entity.device)
            sub_edge_index, sub_edge_attr = subgraph(
                entity_neighbors_all, edge_index, edge_attr, relabel_nodes=True,
                num_nodes=self.entity_graph.graph_data.num_nodes
            )
            sub_graph_entity = Data(x=entity_neighbors_all, edge_index=sub_edge_index, edge_attr=sub_edge_attr)
            output_dict.update({
                "entity_vector_neighbors": self.entity_embedding(entity_neighbors_all),
                "entity_mapping": entity_mapping,
                "sub_graph_entity": sub_graph_entity
            })
        return output_dict

    def news_encoder(self, input_feat):
        """
        Encode news using text feature encoder and news attention layer
        :param input_feat: history_nid, candidate_nid; shape = (B, H), (B, C);
        :return: news vector, shape = (B*(H+C), E); news weight, shape = (B*(H+C), F)
        """
        text_features = self.text_feature_encoder(input_feat)  # shape = (B*(H+C), F, E)
        word_vector = text_features["word_vector"]
        news_mask = text_features["news_mask"]
        if not self.use_news_mask:
            news_mask = None
        y = self.news_encode_layer(word_vector, word_vector, word_vector, news_mask)[0]  # shape = (B*(H+C), F, D)
        # y = self.layer_norm(self.dropout_ne(y))
        y = self.dropout_ne(y)
        # add activation function
        # output = self.layer_norm(self.news_layer(y)[0])
        output = self.news_layer(y)[0]
        news_features = {"news_vector": output}
        if self.use_entity_feature:
            news_features.update({
                "entity_vector": self.local_entity_encoder(text_features.get("entity_vector"), None)
            })
        if self.use_entity_graph or self.use_entity_graph_only:
            entity_vector, entity_mapping = text_features["entity_vector_neighbors"], text_features["entity_mapping"]
            graph_vector = self.global_entity_encoder(entity_vector, text_features["sub_graph_entity"].edge_index)
            graph_vector_batch = self.get_mapping_vector(graph_vector, entity_mapping)
            news_features["entity_graph_vector"] = self.global_entity_att(graph_vector_batch, None)[0]
        return news_features

    def user_encoder(self, input_feat):
        local_history_news = input_feat["history_news"]  # shape = (B, H, D)
        candidate_news_vector = self.get_mapping_vector(input_feat["news_vector"], input_feat["candidate_mapping"])
        # shape = (B, C, D)
        batch_size = local_history_news.shape[0]
        user_feature = None
        user_vectors, candidate_vectors = [local_history_news], []
        if self.use_candidate_local_entity or self.use_candidate_entity_graph or self.use_candidate_news_graph:
            candidate_vectors = [candidate_news_vector.reshape(-1, self.embedding_dim)]  # (B*C, D)
        if self.use_entity_feature:
            user_entity = self.get_mapping_vector(input_feat["entity_vector"], input_feat["history_mapping"])
            cand_entity = self.get_mapping_vector(input_feat["entity_vector"], input_feat["candidate_mapping"])
            user_vectors.append(user_entity)
            if self.use_candidate_local_entity:
                candidate_vectors.append(cand_entity.reshape(-1, self.embedding_dim))
            if self.use_local_entity_only:
                user_feature = user_entity
                candidate_news_vector = cand_entity if self.use_candidate_local_entity else candidate_news_vector
        if self.use_entity_graph or self.use_entity_graph_only:
            entity_graph_u = self.get_mapping_vector(input_feat["entity_graph_vector"], input_feat["history_mapping"])
            entity_graph_c = self.get_mapping_vector(input_feat["entity_graph_vector"], input_feat["candidate_mapping"])
            user_vectors.append(entity_graph_u)
            if self.use_candidate_entity_graph:
                candidate_vectors.append(entity_graph_c.reshape(-1, self.embedding_dim))
            if self.use_entity_graph_only:
                user_feature = entity_graph_u
                candidate_news_vector = entity_graph_c if self.use_candidate_entity_graph else candidate_news_vector
        if self.use_news_graph or self.use_news_graph_only:
            graph_vector = self.global_news_encoder(input_feat["news_vector"], input_feat["sub_graph"].edge_index)
            user_graph_vector = self.get_mapping_vector(graph_vector, input_feat["history_mapping"])
            cand_graph_vector = self.get_mapping_vector(graph_vector, input_feat["candidate_mapping"])
            candidate_vectors.append(cand_graph_vector.reshape(-1, self.embedding_dim))
            user_vectors.append(user_graph_vector)
            if self.use_news_graph_only:
                user_feature = user_graph_vector
                candidate_news_vector = cand_graph_vector if self.use_candidate_entity_graph else candidate_news_vector
        if self.use_fused_feature:
            user_feature = torch.stack(user_vectors, dim=2).view(-1, len(user_vectors), self.embedding_dim)
            user_feature = self.fused_attention(user_feature)[0].view(batch_size, -1, self.embedding_dim)
            # shape = (B, F, H, D)
            if self.use_candidate_local_entity or self.use_candidate_entity_graph or self.use_candidate_news_graph:
                candidate_vectors = torch.stack(candidate_vectors, dim=1)  # shape = (B*C, F, D)
                candidate_vectors = self.fused_attention(candidate_vectors)[0]  # shape = (B*C, D)
                candidate_news_vector = candidate_vectors.view(batch_size, -1, self.embedding_dim)  # shape = (B, C, D)

        if self.use_local_news_only or user_feature is None:
            user_feature = local_history_news
        if self.fusion_method:
            user_vector = self.fusion_aggregator(user_vectors)
            out_dict = {"user_vector": user_vector, "user_weight": None, "candidate_news_vector": candidate_news_vector}
        else:
            # user_feature: shape = (B, H, D); candidate_news_vector: shape = (B, C, D)
            user_embed = self.user_encode_layer(user_feature, user_feature, user_feature)[0]  # shape = (B, H, D)
            out = self.user_layer(user_embed)  # additive attention layer: shape = (B, D)
            out_dict = {"user_vector": out[0], "user_weight": out[1], "candidate_news_vector": candidate_news_vector}
        return out_dict

    def predict(self, candidate_news_vector, user_vector):
        """
        prediction logic: use MLP or Dot-product for prediction
        :param candidate_news_vector: shape = (B, C, D)
        :param user_vector: shape = (B, D)
        :return: softmax possibility of click candidate news
        """
        if self.fusion_method:
            score = self.late_fusion(user_vector, candidate_news_vector)
        else:
            score = self.click_predictor(candidate_news_vector, user_vector)
        return score
