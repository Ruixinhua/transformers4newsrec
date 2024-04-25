# -*- coding: utf-8 -*-
# @Author        : Rui
# @Time          : 2024/4/16 12:14
# @Function      : NRMS model for news recommendation system
import torch.nn as nn

from newsrec.model.general import MultiHeadedAttention
from .base import BaseNRS


class NRMSRSModel(BaseNRS):
    """
    Implementation of NRMS model
    Wu, C., Wu, F., Ge, S., Qi, T., Huang, Y., & Xie, X. (2019).
    Neural News Recommendation with Multi-Head Self-Attention. EMNLP.
    """
    def __init__(self, **kwargs):
        # define document embedding dim before inherit super class
        self.head_num, self.head_dim = kwargs.get("head_num", 20), kwargs.get("head_dim", 20)
        self.embedding_dim = kwargs.get("embedding_dim", self.head_num * self.head_dim)
        super().__init__(**kwargs)
        self.news_encode_layer = MultiHeadedAttention(self.head_num, self.head_dim, self.word_embedding.embed_dim)
        self.user_layer_name = kwargs.get("user_layer_name", "mha")
        if self.user_layer_name == "mha":
            self.user_encode_layer = MultiHeadedAttention(self.head_num, self.head_dim, self.embedding_dim)
        elif self.user_layer_name == "gru":
            self.user_encode_layer = nn.GRU(self.embedding_dim, self.embedding_dim, batch_first=True)
        else:
            raise ValueError(f"Invalid user layer name: {self.user_layer_name}")

    def news_encoder(self, input_feat):
        """
        Encode news using text feature encoder and news attention layer
        :param input_feat: history_nid, candidate_nid; shape = (B, H), (B, C);
        :return: news vector, shape = (B*(H+C), E); news weight, shape = (B*(H+C), F)
        """
        word_vector, news_mask = self.text_feature_encoder(input_feat)  # shape = (B*(H+C), F, E)
        y = self.news_encode_layer(word_vector, word_vector, word_vector)[0]  # shape = (B*(H+C), F, D)
        y = self.dropout_ne(y)
        # add activation function
        output = self.news_layer(y)
        return {"news_vector": output[0], "news_weight": output[1]}

    def user_encoder(self, input_feat):
        x = input_feat["history_news"]  # shape = (B, H, D)
        if self.user_layer_name == "mha":  # the methods used by NRMS original paper
            y = self.user_encode_layer(x, x, x)[0]  # shape = (B, H, D)
            y = self.user_layer(y)  # additive attention layer: shape = (B, D)
            return {"user_vector": y[0], "user_weight": y[1]}
        elif self.user_layer_name == "gru":
            y = self.user_encode_layer(x)[0]  # shape = (B, H, D)
            y = self.user_layer(y)  # additive attention layer: shape = (B, D)
            return {"user_vector": y[0], "user_weight": y[1]}
        else:
            raise ValueError(f"Invalid user layer name: {self.user_layer_name}")
