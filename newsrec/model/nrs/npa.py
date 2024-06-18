# -*- coding: utf-8 -*-
# @Author        : Rui
# @Time          : 2024/4/24 14:20
# @Function      : NPA model for news recommendation system
import os
import torch
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

    def build_input_feat(self, input_feat):
        history_nid, history_mask = input_feat["history_nid"], input_feat["history_mask"]
        candidate_nid, candidate_mask = input_feat["candidate_nid"], input_feat["candidate_mask"]
        # Calculate the count of non-zero values in each row of history_nid
        # Expand uid tensor based on the counts
        uid_history = input_feat["uid"].repeat_interleave(history_mask.sum(dim=1))
        uid_candidate = input_feat["uid"].repeat_interleave(candidate_mask.sum(dim=1))
        input_feat["uid_expand"] = torch.cat((uid_history, uid_candidate))
        history_selected = torch.masked_select(history_nid, history_mask)  # select history based on mask
        candidate_selected = torch.masked_select(candidate_nid, candidate_mask)  # select candidate based on mask
        # get news id in batch of history and candidate
        input_feat["nid"] = torch.cat((history_selected, candidate_selected))
        input_feat["history_mapping"] = torch.full_like(history_nid, -1)
        device = history_nid.device
        history_indices = torch.arange(len(history_selected), dtype=torch.int32).to(device)
        input_feat["history_mapping"].masked_scatter_(history_mask, history_indices)
        input_feat["candidate_mapping"] = torch.full_like(candidate_nid, -1)
        candidate_indices = len(history_selected) + torch.arange(len(candidate_selected), dtype=torch.int32).to(device)
        input_feat["candidate_mapping"].masked_scatter_(candidate_mask, candidate_indices)
        return input_feat

    def news_encoder(self, input_feat):
        """
        Encode news using text feature encoder and news attention layer
        :param input_feat: history_nid, candidate_nid; shape = (B, H), (B, C); uid_expand, (B*(H+C),)
        :return: news vector, shape = (B*(H+C), E); news weight, shape = (B*(H+C), F)
        """
        word_vector, news_mask = self.text_feature_encoder(input_feat)
        news_emb = self.dropout_ne(self.news_encode_layer(word_vector.transpose(1, 2)).transpose(1, 2))
        user_emb = torch.relu(self.transform_news(self.uid_embedding(input_feat["uid_expand"])))
        news_vector, news_weight = self.news_att_layer(news_emb, user_emb)
        return {"news_vector": news_vector, "news_weight": news_weight}

    def user_encoder(self, input_feat):
        news_emb = input_feat["history_news"]
        user_emb = torch.relu(self.transform_user(self.uid_embedding(input_feat["uid"])))
        user_vector, user_weight = self.user_att_layer(news_emb, user_emb)
        return {"user_vector": user_vector, "user_weight": user_weight}
