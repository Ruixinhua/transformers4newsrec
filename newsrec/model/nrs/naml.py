# -*- coding: utf-8 -*-
# @Author        : Rui
# @Time          : 2024/4/23 21:27
# @Function      : NAML model for news recommendation system
import torch
import torch.nn as nn
import torch.nn.functional as F

from newsrec.model.general import AttLayer, Conv1D
from .base import BaseNRS
from newsrec.utils import reshape_tensor


class NAMLRSModel(BaseNRS):
    def __init__(self, **kwargs):
        self.use_category = True  # use category and subvert for NAML model
        super(NAMLRSModel, self).__init__(**kwargs)
        self.num_filters, self.window_size = kwargs.get("num_filters", 300), kwargs.get("window_size", 3)
        if "category" in self.cat_feature:
            self.category_affine = nn.Linear(self.category_dim, self.num_filters, bias=True)
        if "subvert" in self.cat_feature:
            self.subvert_affine = nn.Linear(self.category_dim, self.num_filters, bias=True)
        if "title" in self.text_feature:
            self.title_cnn = Conv1D(self.embedding_dim, self.num_filters, self.window_size)
            self.title_att = AttLayer(self.num_filters, self.attention_hidden_dim)
        if "abstract" in self.text_feature:
            self.abstract_cnn = Conv1D(self.embedding_dim, self.num_filters, self.window_size)
            self.abstract_att = AttLayer(self.num_filters, self.attention_hidden_dim)
        if "body" in self.text_feature:
            self.body_cnn = Conv1D(self.embedding_dim, self.num_filters, self.window_size)
            self.body_att = AttLayer(self.num_filters, self.attention_hidden_dim)
        self.multi_att = AttLayer(self.num_filters, self.attention_hidden_dim)
        self.news_layer = None  # set to None for NAML model

    def news_encoder(self, input_feat):
        """input_feat contains: news title, body, category, and subvert"""
        # 1. word embedding: [batch_size * news_num, max_news_length, word_embedding_dim]
        news_all_feature = self.feature_embedding(input_feat["nid"])  # shape = (B, H+C, F)
        news_feature_dict = self.feature_embedding.select_feature(news_all_feature)
        news_vector = []
        for feature in self.text_feature:
            feature_tokens = news_feature_dict[feature]
            feature_mask = torch.where(feature_tokens == self.pad_token_id, 0, 1)
            # feature_tokens: shape = (B, H+C, F), F is the sum of feature used in news
            if len(feature_tokens.shape) == 3:
                feature_tokens = reshape_tensor(feature_tokens)  # out: (B * (H+C), F)
            feature_vector = self.dropout_we(self.word_embedding(feature_tokens, feature_mask))
            feature_cnn, feature_att = getattr(self, f"{feature}_cnn"), getattr(self, f"{feature}_att")
            feature_vector = self.dropout_ne(feature_cnn(feature_vector.transpose(1, 2)).transpose(1, 2))
            # 3. attention layer: [batch_size * news_num, kernel_num]
            news_vector.append(feature_att(feature_vector)[0])
            # shape = (B*(H+C), F, E)
        # 2. cat embedding: [batch_size * news_num, kernel_num]
        for feature in self.cat_feature:
            affine, embedding = getattr(self, f"{feature}_affine"), getattr(self, f"{feature}_embedding")
            vector = F.relu(affine(embedding(news_feature_dict[feature])), inplace=True)
            if len(vector.shape) == 3:
                vector = reshape_tensor(vector)
            news_vector.append(vector)
        news_vector = torch.stack(news_vector, dim=1)
        # 2. multi-view attention: [batch_size*news_num, n, kernel_num]
        news_vector, weight = self.multi_att(news_vector)
        return {"news_vector": news_vector, "news_weight": weight}
