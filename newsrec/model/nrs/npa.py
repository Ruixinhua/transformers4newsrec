# -*- coding: utf-8 -*-
# @Author        : Rui
# @Time          : 2024/4/24 14:20
# @Function      : NPA model for news recommendation system
import os
import torch.nn.functional as F
import torch.nn as nn

from newsrec.model.general import PersonalizedAttentivePooling
from .base import BaseNRS


class NPARSModel(BaseNRS):
    """
    Implementation of NPM model
    Wu, Chuhan et al. “NPA: Neural News Recommendation with Personalized Attention.”
    Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (2019): n. pag.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_filters = kwargs.get("num_filters", 300)
        self.uid_embed_dim, self.window_size = kwargs.get("uid_embed_dim", 100), kwargs.get("window_size", 3)
        padding = (self.window_size - 1) // 2
        assert 2 * padding == self.window_size - 1, "Kernel size must be an odd number"
        self.news_encode_layer = nn.Sequential(
            nn.Conv1d(self.embedding_dim, self.num_filters, self.window_size, padding=padding),
            nn.ReLU(inplace=True)
        )
        self.uid_embedding = nn.Embedding(len(self.user_history) + 1, self.uid_embed_dim)
        self.transform_news = nn.Linear(self.uid_embed_dim, self.attention_hidden_dim)
        self.transform_user = nn.Linear(self.uid_embed_dim, self.attention_hidden_dim)
        self.news_att_layer = PersonalizedAttentivePooling(self.num_filters, self.attention_hidden_dim)
        self.user_att_layer = PersonalizedAttentivePooling(self.num_filters, self.attention_hidden_dim)

    def news_encoder(self, input_feat):
        """input_feat: Size is [N * H, S]"""
        word_vector, news_mask = self.text_feature_encoder(input_feat)
        expand_shape = input_feat["history_nid"].shape[1]+input_feat["candidate_nid"].shape[1]
        uid_expand = input_feat["uid"].repeat_interleave(expand_shape)
        news_emb = self.dropout_ne(self.news_encode_layer(word_vector.transpose(1, 2)).transpose(1, 2))
        user_emb = F.relu(self.transform_news(self.uid_embedding(uid_expand)), inplace=True)
        news_vector, news_weight = self.news_att_layer(news_emb, user_emb)
        return {"news_vector": news_vector, "news_weight": news_weight}

    def user_encoder(self, input_feat):
        news_emb = input_feat["history_news"]
        user_emb = F.relu(self.transform_user(self.uid_embedding(input_feat["uid"])), inplace=True)
        user_vector, user_weight = self.user_att_layer(news_emb, user_emb)
        return {"user_vector": user_vector, "user_weight": user_weight}
