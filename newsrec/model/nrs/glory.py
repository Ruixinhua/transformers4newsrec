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
        self.use_entity = kwargs.get("entity_feature") is not None  # whether to use entity feature
        self.use_entity_only = kwargs.get("use_entity_only", False)
        self.use_local_news_only = kwargs.get("use_local_news_only", False)
        self.use_entity_graph = kwargs.get("use_entity_graph", False)
        self.use_entity_graph_only = kwargs.get("use_entity_graph_only", False)
        self.use_news_graph = kwargs.get("use_news_graph", False)
        self.use_news_graph_only = kwargs.get("use_news_graph_only", False)
        self.use_fused_feature = kwargs.get("use_fused_feature", False)
        self.apply_candidate_feature = kwargs.get("apply_candidate_feature", False)
        # check if there are more than two conditions of "only" flag is true
        only = [self.use_entity_only, self.use_local_news_only, self.use_entity_graph_only, self.use_news_graph_only]
        if sum(only) > 1:
            raise ValueError(
                f"Only one of the following flags can be set to True: "
                f"use_local_news_only: {self.use_local_news_only}; use_news_graph_only: {self.use_news_graph_only}; "
                f"use_entity_only: {self.use_entity_only}; use_entity_graph_only: {self.use_entity_graph_only}")
        self.news_encode_layer = MultiHeadedAttention(self.head_num, self.head_dim, word_embed_dim,
                                                      use_flash_att=self.use_flash_att)
        self.user_layer_name = kwargs.get("user_layer_name", "mha")
        if self.user_layer_name == "mha":
            self.user_encode_layer = MultiHeadedAttention(self.head_num, self.head_dim, self.embedding_dim,
                                                          use_flash_att=self.use_flash_att)
        if self.use_news_graph or self.use_news_graph_only:
            self.news_graph = load_news_graph(**kwargs)
            news_neighbors_num = kwargs.get("news_neighbors_num", 8)
            self.news_neighbor_mapper = self.news_graph.neighbor_mapper
            neighbor_matrix = torch.zeros(len(self.news_neighbor_mapper) + 1, news_neighbors_num, dtype=torch.int32)
            for k, v in self.news_neighbor_mapper.items():
                neighbor_matrix[k, :min(news_neighbors_num, len(v))] = torch.tensor(v[:news_neighbors_num])
            self.news_neighbors = nn.Embedding.from_pretrained(neighbor_matrix, freeze=True)
            self.k_hops = kwargs.get("k_hops", 2)
            self.global_news_encoder = Sequential("x,index", [
                (GatedGraphConv(self.embedding_dim, num_layers=3, aggr="add"), "x,index -> x"),
            ])
            if self.use_layernorm:
                self.graph_layer_norm = nn.LayerNorm(self.embedding_dim)

        if self.use_entity_graph or self.use_entity_graph_only:
            self.entity_graph = load_entity_graph(**kwargs)
            entity_neighbors_num = kwargs.get("entity_neighbors_num", 8)
            self.entity_neighbor_mapper = self.entity_graph.neighbor_mapper
            neighbor_matrix = torch.zeros(len(self.entity_neighbor_mapper)+1, entity_neighbors_num, dtype=torch.int32)
            for k, v in self.entity_neighbor_mapper.items():
                neighbor_matrix[k, :min(entity_neighbors_num, len(v))] = torch.tensor(v[:entity_neighbors_num])
            self.entity_neighbors = nn.Embedding.from_pretrained(neighbor_matrix, freeze=True)

            self.global_entity_graph_encoder = Sequential('x, mask', [
                (self.entity_embedding, 'x -> x'),
                (nn.Dropout(p=kwargs.get("dropout_eg", 0.2)), 'x -> x'),
                (MultiHeadedAttention(self.head_num, self.head_dim, self.entity_dim,
                                      use_flash_att=self.use_flash_att), 'x,x,x,mask -> x,x_att'),
                # nn.LayerNorm(self.embedding_dim),
                # nn.Dropout(p=cfg.dropout_probability),
                (AttLayer(self.head_num * self.head_dim, self.attention_hidden_dim), "x,mask -> x,x_att"),
                # nn.LayerNorm(cfg.model.head_num * cfg.model.head_dim),
            ])

        if self.use_entity or self.use_entity_only:
            self.entity_graph = load_entity_graph(**kwargs)
            self.local_entity_encoder = Sequential("x, mask", [
                (nn.Dropout(p=kwargs.get("dropout_le", 0.2)), "x -> x"),
                (MultiHeadedAttention(int(self.entity_dim / self.head_dim), self.head_dim, self.entity_dim,
                                      use_flash_att=self.use_flash_att),
                 "x,x,x,mask -> x,x_att"),
                # nn.LayerNorm(self.entity_dim),
                (nn.Dropout(p=kwargs.get("dropout_le", 0.2)), "x -> x"),
                (AttLayer(self.entity_dim, self.attention_hidden_dim), "x,mask -> x,x_att"),
                # nn.LayerNorm(self.entity_dim),
                (nn.Linear(self.entity_dim, self.embedding_dim), "x -> x"),
                nn.LeakyReLU(0.2),
            ])

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
            history_neighbor_all = current
            for _ in range(self.k_hops):  # get the neighbors of the history news and append to all list
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
        if self.use_entity:
            entity = news_feature_dict["entity"]
            entity_vector = self.entity_embedding(entity)
            output_dict.update({"entity_vector": entity_vector})
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
        if self.use_entity:
            entity_vector = self.local_entity_encoder(text_features.get("entity_vector"), None)
            news_features.update({"entity_vector": entity_vector})
        return news_features

    def user_encoder(self, input_feat):
        local_history_news = input_feat["history_news"]  # shape = (B, H, D)
        candidate_news_vector = self.get_mapping_vector(input_feat["news_vector"], input_feat["candidate_mapping"])
        # shape = (B, C, D)
        batch_size = local_history_news.shape[0]
        user_vectors, candidate_vectors = [], []
        user_feature = None
        if self.use_fused_feature:
            user_vectors = [local_history_news]
            if self.apply_candidate_feature:
                candidate_vectors = [candidate_news_vector.reshape(-1, self.embedding_dim)]  # (B*C, D)
        if self.use_entity or self.use_entity_only:
            user_entity = self.get_mapping_vector(input_feat["entity_vector"], input_feat["history_mapping"])
            cand_entity = self.get_mapping_vector(input_feat["entity_vector"], input_feat["candidate_mapping"])
            if self.use_fused_feature:
                user_vectors.append(user_entity)
                if self.apply_candidate_feature:
                    candidate_vectors.append(cand_entity.reshape(-1, self.embedding_dim))
            if self.use_entity_only:
                user_feature = user_entity
                candidate_news_vector = cand_entity if self.apply_candidate_feature else candidate_news_vector
        if self.use_news_graph or self.use_news_graph_only:
            graph_vector = self.global_news_encoder(input_feat["news_vector"], input_feat["sub_graph"].edge_index)
            user_graph_vector = self.get_mapping_vector(graph_vector, input_feat["history_mapping"])
            cand_graph_vector = self.get_mapping_vector(graph_vector, input_feat["candidate_mapping"])
            if self.use_fused_feature:
                candidate_vectors.append(cand_graph_vector.reshape(-1, self.embedding_dim))
                user_vectors.append(user_graph_vector)
            if self.use_news_graph_only:
                user_feature = user_graph_vector
                candidate_news_vector = cand_graph_vector if self.apply_candidate_feature else candidate_news_vector
        if self.use_fused_feature:
            user_feature = torch.stack(user_vectors, dim=2).view(-1, len(user_vectors), self.embedding_dim)
            user_feature = self.fused_attention(user_feature)[0].view(batch_size, -1, self.embedding_dim)
            # shape = (B, F, H, D)
            if self.apply_candidate_feature:
                candidate_vectors = torch.stack(candidate_vectors, dim=1)  # shape = (B*C, F, D)
                candidate_vectors = self.fused_attention(candidate_vectors)[0]  # shape = (B*C, D)
                candidate_news_vector = candidate_vectors.view(batch_size, -1, self.embedding_dim)  # shape = (B, C, D)
        if self.use_local_news_only or user_feature is None:
            user_feature = local_history_news
        # user_feature: shape = (B, H, D); candidate_news_vector: shape = (B, C, D)
        user_embed = self.user_encode_layer(user_feature, user_feature, user_feature)[0]  # shape = (B, H, D)
        output = self.user_layer(user_embed)  # additive attention layer: shape = (B, D)
        return {"user_vector": output[0], "user_weight": output[1], "candidate_news_vector": candidate_news_vector}
