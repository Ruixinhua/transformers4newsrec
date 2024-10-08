# -*- coding: utf-8 -*-
# @Author        : Rui
# @Time          : 2024/4/29 19:28
# @Function      : Implement the model of GLORY
import copy

import torch
import torch.nn as nn
from newsrec.data import load_news_graph, load_entity_graph
from newsrec.utils import reshape_tensor
from newsrec.model.general import LateFusion, FusionAggregator, BATMEncoder, MHAEncoder, AttLayerEncoder, GraphEncoder
from torch_geometric.nn import Sequential
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
        default_args = {
            "head_num": 20, "head_dim": 20, "topic_num": 60, "topic_dim": 30, "topic_layer_name": "base_att",
            "attention_hidden_dim": 200, "window_size": 3, "num_filters": 300, "use_news_mask": False,
            "use_local_entity": False, "use_local_entity_only": False, "use_local_news_only": False,
            "use_entity_graph": False, "use_entity_graph_only": False, "use_news_graph": False, "use_flash_att": False,
            "use_news_graph_only": False, "use_candidate_local_entity": False, "use_candidate_entity_graph": False,
            "use_candidate_news_graph": False, "use_fused_feature": False,
            "fusion_method": "weighted", "use_same_fusion": False,  # weighted, max, avg; for user and candidate fusion
            "aggregator_method": "mlp",  # mlp/att: to aggregate user and candidate features
            "local_news_encoder_name": "batm_att",  # batm_att, mha_att
            "use_user_mask": False, "user_encoder_name": "mha_att",  # mha_att, batm_att
            "gnn_model": "GGNN",  # GGNN, GAT, GATConv, GraphSAGE
            "dropout_ne": 0.2, "dropout_ue": 0.2, "dropout_eg": 0.2, "dropout_le": 0.2,
            "k_hops_ng": 2, "k_hops_eg": 2, "entity_neighbors_num": 8, "news_neighbors_num": 8,
        }
        for key, value in default_args.items():
            kwargs.setdefault(key, value)
        super().__init__(**kwargs)
        word_embed_dim = self.word_embedding.embed_dim
        self.use_local_entity, self.use_local_entity_only = kwargs["use_local_entity"], kwargs["use_local_entity_only"]
        self.use_local_news_only = kwargs["use_local_news_only"]
        self.use_entity_graph, self.use_entity_graph_only = kwargs["use_entity_graph"], kwargs["use_entity_graph_only"]
        self.use_news_graph, self.use_news_graph_only = kwargs["use_news_graph"], kwargs["use_news_graph_only"]
        self.use_candidate_local_entity = kwargs["use_candidate_local_entity"]
        self.use_candidate_entity_graph = kwargs["use_candidate_entity_graph"]
        self.use_candidate_news_graph = kwargs["use_candidate_news_graph"]
        self.use_fused_feature = kwargs["use_fused_feature"]
        self.fusion_method, self.aggregator_method = kwargs["fusion_method"], kwargs["aggregator_method"]
        self.gnn_model, self.local_news_encoder_name = kwargs["gnn_model"], kwargs["local_news_encoder_name"]
        self.user_encoder_name = kwargs["user_encoder_name"]
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
        if self.local_news_encoder_name.lower() == "mha_att":
            self.local_news_encoder = Sequential("x,mask", [
                (nn.Dropout(p=kwargs["dropout_ne"]), "x -> x"),
                (MHAEncoder(input_feat_dim=word_embed_dim, head_num=kwargs["head_num"],
                            head_dim=kwargs["head_dim"], use_flash_att=self.use_flash_att), "x,mask -> x,x_att"),
                (AttLayerEncoder(kwargs["head_num"]*kwargs["head_dim"], self.attention_hidden_dim), "x,mask -> x,x_att")
            ])
        elif self.local_news_encoder_name.lower() == "batm_att":
            self.local_news_encoder = Sequential("x,mask", [
                (nn.Dropout(p=kwargs["dropout_ne"]), "x -> x"),
                (BATMEncoder(input_feat_dim=word_embed_dim, **kwargs), "x,mask -> x,x_att"),
                (AttLayerEncoder(word_embed_dim, self.attention_hidden_dim), "x,mask -> x,x_att")
            ])
        elif self.local_news_encoder_name.lower() == "batm_mha_att":
            self.local_news_encoder = Sequential("x,mask", [
                (nn.Dropout(p=kwargs["dropout_ne"]), "x -> x"),
                (BATMEncoder(input_feat_dim=word_embed_dim, **kwargs), "x,mask -> x,x_att"),
                (MHAEncoder(input_feat_dim=word_embed_dim, head_num=kwargs["head_num"],
                            head_dim=kwargs["head_dim"], use_flash_att=self.use_flash_att), "x,mask -> x,x_att"),
                (AttLayerEncoder(kwargs["head_num"]*kwargs["head_dim"], self.attention_hidden_dim), "x,mask -> x,x_att")
            ])
        else:
            raise ValueError(f"Unknown local news encoder: {self.local_news_encoder_name}")
        self.encoding_dim = self.local_news_encoder[-1].output_feat_dim  # news encoding dimension
        # local news encoder
        if self.user_encoder_name == "mha_att":
            self.local_user_encoder = Sequential("x,mask", [
                (nn.Dropout(p=kwargs["dropout_ue"]), "x -> x"),
                (MHAEncoder(input_feat_dim=self.encoding_dim, head_num=kwargs["head_num"], head_dim=kwargs["head_dim"],
                            output_feat_dim=self.encoding_dim, use_flash_att=self.use_flash_att), "x,mask -> x,x_att"),
                (AttLayerEncoder(self.encoding_dim, self.attention_hidden_dim), "x,mask -> x,x_att")
            ])
        elif self.user_encoder_name == "batm_att":
            self.local_user_encoder = Sequential("x,mask", [
                (nn.Dropout(p=kwargs["dropout_ue"]), "x -> x"),
                (BATMEncoder(input_feat_dim=self.encoding_dim, **kwargs), "x,mask -> x,x_att"),
                (AttLayerEncoder(self.encoding_dim, self.attention_hidden_dim), "x,mask -> x,x_att")
            ])
        else:
            raise ValueError(f"Unknown user encoder: {self.user_encoder_name}")
        # global news graph encoder
        if self.use_news_graph or self.use_news_graph_only:
            self.news_graph = load_news_graph(**kwargs)
            self.k_hops_ng, self.news_neighbors_num = kwargs["k_hops_ng"], kwargs["news_neighbors_num"]
            self.global_news_encoder = GraphEncoder(
                self.news_graph, self.news_neighbors_num, gnn_model=self.gnn_model,
                input_feat_dim=self.encoding_dim, output_feat_dim=self.encoding_dim
            )
        self.use_entity_feature = ((self.entity_feature and len(self.entity_feature)) and
                                   (self.use_local_entity or self.use_local_entity_only or
                                    self.use_entity_graph or self.use_entity_graph_only))
        # local entity encoder
        if self.use_entity_feature:
            self.local_entity_encoder = Sequential("x,mask", [
                (nn.Dropout(p=kwargs["dropout_le"]), "x -> x"),
                (MHAEncoder(input_feat_dim=self.entity_dim, head_num=int(self.entity_dim / kwargs["head_dim"]),
                            head_dim=kwargs["head_dim"], use_flash_att=self.use_flash_att),
                 "x,mask -> x,x_att"),
                (nn.Dropout(p=kwargs["dropout_le"]), "x -> x"),
                (AttLayerEncoder(self.entity_dim, self.attention_hidden_dim), "x,mask -> x,x_att"),
                (nn.Linear(self.entity_dim, self.encoding_dim), "x -> x"),
                nn.LeakyReLU(0.2),
            ])
        # global entity graph encoder
        if self.use_entity_graph or self.use_entity_graph_only:
            self.entity_graph = load_entity_graph(**kwargs)
            self.k_hops_eg, self.entity_neighbors_num = kwargs["k_hops_eg"], kwargs["entity_neighbors_num"]
            self.global_entity_encoder = GraphEncoder(
                self.entity_graph, self.entity_neighbors_num, gnn_model=self.gnn_model,
                input_feat_dim=self.entity_dim, output_feat_dim=self.entity_dim
            )
            self.global_entity_att = Sequential('x, mask', [
                (nn.Dropout(p=kwargs.get("dropout_eg", 0.2)), 'x -> x'),
                (MHAEncoder(input_feat_dim=self.entity_dim, head_num=kwargs["head_num"], head_dim=kwargs["head_dim"],
                            use_flash_att=self.use_flash_att, output_feat_dim=self.encoding_dim), 'x,mask -> x,x_att'),
                (AttLayerEncoder(self.encoding_dim, self.attention_hidden_dim), "x,mask -> x,x_att"),
            ])
        # fusion method
        if self.fusion_method:
            # late fusion layer: Input_shape = List of tensors (B, F, D), Output_shape = (B, F, D)
            self.late_fusion = LateFusion(self.fusion_method, self.encoding_dim)
            user_feature_num = sum([self.use_local_entity, self.use_entity_graph, self.use_news_graph]) + 1
            self.user_fusion_aggregator = FusionAggregator(
                self.encoding_dim, user_feature_num, kwargs["attention_hidden_dim"], self.aggregator_method
            )
            cand_feature_num = sum([self.use_candidate_local_entity, self.use_candidate_entity_graph,
                                    self.use_candidate_news_graph]) + 1
            if kwargs["use_same_fusion"] and user_feature_num == cand_feature_num:
                self.cand_fusion_aggregator = self.user_fusion_aggregator
            else:
                if kwargs["use_same_fusion"]:
                    print("The number of user and candidate features are different. Use different fusion network.")
                self.cand_fusion_aggregator = FusionAggregator(
                    self.encoding_dim, cand_feature_num, kwargs["attention_hidden_dim"], self.aggregator_method
                )

        # fused attention layer
        if self.use_fused_feature:
            self.fused_attention = AttLayerEncoder(self.encoding_dim, self.attention_hidden_dim)

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
                current = self.global_news_encoder.neighbors(current)  # get current history news neighbors
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
        if len(news_tokens.shape) == 3:
            news_tokens = reshape_tensor(news_tokens)  # out: (B * (H+C), F)
        # news_tokens: shape = (B*(H+C), F), F is the sum of feature used in news
        news_mask = torch.where(news_tokens == self.pad_token_id, 0, 1).to(news_tokens.device)  # shape = (B*(H+C), F)
        word_vector = self.dropout_we(self.word_embedding(news_tokens, news_mask))  # shape = (B*(H+C), F, E)
        output_dict = {"word_vector": word_vector, "news_mask": news_mask}
        if self.use_entity_feature:
            entity_tokens = news_feature_dict["entity"]
            if len(entity_tokens.shape) == 3:
                entity_tokens = reshape_tensor(entity_tokens)
            output_dict.update({"entity_vector": self.entity_embedding(entity_tokens)})
        if self.use_entity_graph or self.use_entity_graph_only:
            entity = torch.masked_select(news_feature_dict["entity"], news_feature_dict["entity"] != 0)
            entity_neighbors_all = copy.deepcopy(entity)
            for _ in range(self.k_hops_eg):
                entity = self.global_entity_encoder.neighbors(entity)
                entity_neighbors_all = torch.cat([entity_neighbors_all, torch.masked_select(entity, entity != 0)])
            entity_neighbors_all = torch.unique(entity_neighbors_all)
            entity_mapping, = self.get_mapping_index(entity_neighbors_all, news_feature_dict["entity"])
            if len(entity_mapping.shape) == 3:
                entity_mapping = reshape_tensor(entity_mapping)
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
        news_vector, _ = self.local_news_encoder(word_vector, news_mask)
        news_features = {"news_vector": news_vector}
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
        """
        Encode user using news vector and user attention layer
        :param input_feat:
        :return:
        user vector, shape = (B, E) or (B, H, E) for fusion method
        candidate news vector, shape = (B, C, E)
        """
        local_history_news = input_feat["history_news"]  # shape = (B, H, D)
        candidate_news_vector = self.get_mapping_vector(input_feat["news_vector"], input_feat["candidate_mapping"])
        # shape = (B, C, D)
        batch_size = local_history_news.shape[0]
        user_mask = None
        if self.use_user_mask:
            user_mask = input_feat["history_mask"]
        user_feature = None
        user_vectors, candidate_vectors = [local_history_news], [candidate_news_vector.reshape(-1, self.encoding_dim)]
        if self.use_local_entity or self.use_local_entity_only:
            user_entity = self.get_mapping_vector(input_feat["entity_vector"], input_feat["history_mapping"])
            cand_entity = self.get_mapping_vector(input_feat["entity_vector"], input_feat["candidate_mapping"])
            user_vectors.append(user_entity)
            if self.use_candidate_local_entity:
                candidate_vectors.append(cand_entity.reshape(-1, self.encoding_dim))
            if self.use_local_entity_only:
                user_feature = user_entity
                candidate_news_vector = cand_entity if self.use_candidate_local_entity else candidate_news_vector
        if self.use_entity_graph or self.use_entity_graph_only:
            entity_graph_u = self.get_mapping_vector(input_feat["entity_graph_vector"], input_feat["history_mapping"])
            entity_graph_c = self.get_mapping_vector(input_feat["entity_graph_vector"], input_feat["candidate_mapping"])
            user_vectors.append(entity_graph_u)
            if self.use_candidate_entity_graph:
                candidate_vectors.append(entity_graph_c.reshape(-1, self.encoding_dim))
            if self.use_entity_graph_only:
                user_feature = entity_graph_u
                candidate_news_vector = entity_graph_c if self.use_candidate_entity_graph else candidate_news_vector
        if self.use_news_graph or self.use_news_graph_only:
            graph_vector = self.global_news_encoder(input_feat["news_vector"], input_feat["sub_graph"].edge_index)
            user_graph_vector = self.get_mapping_vector(graph_vector, input_feat["history_mapping"])
            cand_graph_vector = self.get_mapping_vector(graph_vector, input_feat["candidate_mapping"])
            user_vectors.append(user_graph_vector)
            if self.use_candidate_news_graph:
                candidate_vectors.append(cand_graph_vector.reshape(-1, self.encoding_dim))
            if self.use_news_graph_only:
                user_feature = user_graph_vector
                candidate_news_vector = cand_graph_vector if self.use_candidate_news_graph else candidate_news_vector
        if self.use_fused_feature:
            user_feature = torch.stack(user_vectors, dim=2).view(-1, len(user_vectors), self.encoding_dim)
            user_feature = self.fused_attention(user_feature)[0].view(batch_size, -1, self.encoding_dim)
            # shape = (B, F, H, D)
            if self.use_candidate_local_entity or self.use_candidate_entity_graph or self.use_candidate_news_graph:
                candidate_vectors = torch.stack(candidate_vectors, dim=1)  # shape = (B*C, F, D)
                candidate_vectors = self.fused_attention(candidate_vectors)[0]  # shape = (B*C, D)
                candidate_news_vector = candidate_vectors.view(batch_size, -1, self.encoding_dim)  # shape = (B, C, D)

        if self.use_local_news_only or user_feature is None:
            user_feature = local_history_news
        if self.fusion_method:
            user_vector = self.user_fusion_aggregator(user_vectors)
            candidate_vectors = [vector.view(batch_size, -1, self.encoding_dim) for vector in candidate_vectors]
            candidate_news_vector = self.cand_fusion_aggregator(candidate_vectors)
        else:
            # user_feature: shape = (B, H, D); candidate_news_vector: shape = (B, C, D)
            user_vector, _ = self.local_user_encoder(user_feature, user_mask)
        return {"user_vector": user_vector, "candidate_news_vector": candidate_news_vector}

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
