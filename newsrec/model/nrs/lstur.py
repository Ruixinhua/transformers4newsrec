# -*- coding: utf-8 -*-
# @Author        : Rui
# @Time          : 2024/4/23 9:18
# @Function      : Define the LSTUR model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from newsrec.model.general import AttLayer, Conv1D, FrozenEmbedding
from newsrec.utils import reshape_tensor
from .base import BaseNRS


class LSTURRSModel(BaseNRS):
    """
    Implementation of LSTRU model
    Ref: An, Mingxiao et al. “Neural News Recommendation with Long- and Short-term User Representations.” ACL (2019).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.masking_probability = 1.0 - kwargs.get("long_term_masking_probability", 0.1)
        self.use_category = kwargs.get("use_category", False)
        self.num_filters = kwargs.get("num_filters", 300)
        self.window_size = kwargs.get("window_size", 3)
        self.user_embed_method = kwargs.get("user_embed_method", "concat")
        padding = (self.window_size - 1) // 2
        assert 2 * padding == self.window_size - 1, "Kernel size must be an odd number"
        self.news_encode_layer = Conv1D(self.embedding_dim, self.num_filters, self.window_size)
        self.news_att_layer = AttLayer(self.num_filters, self.attention_hidden_dim)  # output size => [N, num_filters]
        news_dim = self.num_filters
        uid_embed_dim = kwargs.get("uid_embed_dim", 100)
        if self.cat_feature and len(self.cat_feature):
            news_dim = self.num_filters + self.category_dim * len(self.cat_feature)
        if self.user_embed_method == "init" or self.user_embed_method == "concat":
            self.uid_embedding = nn.Embedding(len(self.user_history) + 1, uid_embed_dim)
            self.user_affine = nn.Linear(uid_embed_dim, news_dim)
        self.user_encode_layer = nn.GRU(news_dim, news_dim, batch_first=True, bidirectional=False)
        if self.user_embed_method == "concat":
            self.user_affine = None
            self.transform_layer = nn.Linear(news_dim + uid_embed_dim, news_dim)
        self.user_layer = None  # no attentive layer for LSTUR model
        self.embedding_dim = news_dim
        self.news_embedding = FrozenEmbedding(len(self.feature_embedding), news_dim)  # news_num, embed_dim
        self.user_embedding = FrozenEmbedding(len(self.user_history), news_dim)  # user_num, embed_dim

    def cat_feature_encoder(self, input_feat):
        # concat history news and candidate news
        if not self.cat_feature and not len(self.cat_feature):
            raise ValueError("cat_feature is not set in the model")
        news_all_feature = self.feature_embedding(input_feat["nid"])  # shape = (B, H+C, F)
        cat_dict = self.feature_embedding.select_feature(news_all_feature)
        cat_embedding = self.category_embedding(cat_dict["category"]) if "category" in self.cat_feature else None
        sub_embedding = self.subvert_embedding(cat_dict["subvert"]) if "subvert" in self.cat_feature else None
        if cat_embedding is not None and sub_embedding is not None:
            cat_vector = torch.cat([cat_embedding, sub_embedding], dim=-1)
        elif cat_embedding is not None:
            cat_vector = cat_embedding
        elif sub_embedding is not None:
            cat_vector = sub_embedding
        else:
            raise ValueError("wrong category feature passing!")
        return cat_vector

    def news_encoder(self, input_feat):
        """input_feat: Size is [N * H, S]"""
        # 1. worod embedding
        word_vector, news_mask = self.text_feature_encoder(input_feat)  # shape = (B*(H+C), F, E)
        # 2. CNN
        cnn_features = self.news_encode_layer(word_vector.transpose(1, 2)).transpose(1, 2)
        # 3. attention
        y, weight = self.news_att_layer(self.dropout_ne(cnn_features))  # y.shape = (B*(H+C), D)
        if self.cat_feature and len(self.cat_feature):
            cat_vector = self.cat_feature_encoder(input_feat)
            if len(cat_vector.shape) == 3:
                cat_vector = reshape_tensor(cat_vector)
            y = torch.cat([y, self.dropout_ce(cat_vector)], dim=1)  # y.shape = (B*(H+C), WD+CD)
        return {"news_vector": y, "news_weight": weight}

    def user_encoder(self, input_feat):
        history_news, user_ids = input_feat["history_news"], input_feat["uid"]
        history_length = torch.sum(input_feat["history_mask"], dim=-1).cpu()
        history_length[history_length == 0] = 1  # avoid zero history recording
        packed_y = pack_padded_sequence(
            history_news, history_length, batch_first=True, enforce_sorted=False
        )
        if self.user_embed_method == "init":
            user_embed = F.relu(self.user_affine(self.uid_embedding(user_ids)), inplace=True)
            _, user_vector = self.user_encode_layer(
                packed_y, user_embed.unsqueeze(dim=0)
            )
            user_vector = user_vector.squeeze(dim=0)
        elif self.user_embed_method == "concat":
            user_embed = self.uid_embedding(user_ids)
            last_hidden = self.user_encode_layer(packed_y)[1].squeeze(dim=0)
            user_vector = F.relu(
                self.transform_layer(torch.cat((last_hidden, user_embed), dim=1)),
                inplace=True,
            )
        else:  # default use last hidden output from GRU network
            user_vector = self.user_encode_layer(packed_y)[1].squeeze(dim=0)
        return {"user_vector": user_vector}
