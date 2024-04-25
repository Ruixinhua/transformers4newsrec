# -*- coding: utf-8 -*-
# @Author        : Rui
# @Time          : 2024/4/14 19:31
# @Function      : define different layers for news recommendation system
import math

import torch
import torch.nn as nn


class ClickPredictor(nn.Module):
    """
    Click predictor for news recommendation system
    """

    def __init__(self, **kwargs):
        """

        :param kwargs: dnn predictor, input_size, hidden_size; dot_product predictor
        """
        super(ClickPredictor, self).__init__()
        # TODO: is sqrt(input_size) a good default value?
        self.predictor = kwargs.get("predictor", "dot_product")
        if self.predictor == "dnn":
            predictor_input_size = kwargs.get("predictor_input_size")
            predictor_hidden_size = kwargs.get("predictor_hidden_size", int(math.sqrt(predictor_input_size)))
            self.dnn = nn.Sequential(
                nn.Linear(predictor_input_size, predictor_hidden_size), nn.ReLU(),
                nn.Linear(predictor_hidden_size, 1),
            )

    def forward(self, candidate_news_vector, user_vector):
        """

        :param candidate_news_vector: batch_size, candidate_size, X or batch_size, X
        :param user_vector: batch_size, X
        :return: prediction probability, batch_size
        """
        if self.predictor == "dot_product":
            # batch_size, candidate_size
            if len(user_vector.shape) == 3:  # user_vector.shape = (batch_size, candidate_size, X)
                probability = torch.sum(user_vector * candidate_news_vector, dim=-1)
            else:
                probability = torch.bmm(candidate_news_vector, user_vector.unsqueeze(dim=-1)).squeeze(dim=-1)
        else:
            if len(candidate_news_vector.shape) != len(user_vector.shape):
                # expand user_vector to the same shape as candidate_news_vector
                user_vector = torch.unsqueeze(user_vector, 1).expand(
                    [user_vector.shape[0], candidate_news_vector.shape[1], -1]
                )
            probability = self.dnn(torch.cat((candidate_news_vector, user_vector), dim=-1)).squeeze()
        return probability


class AttLayer(nn.Module):

    def __init__(self, word_emb_dim, attention_hidden_dim):
        super().__init__()
        # build attention network
        self.attention = nn.Sequential(
            nn.Linear(word_emb_dim, attention_hidden_dim, bias=True),
            nn.Tanh(),
            nn.Linear(attention_hidden_dim, 1, bias=True),
            nn.Softmax(dim=-2)
        )

    def forward(self, x):
        attention_weight = self.attention(x)
        y = torch.sum(x * attention_weight, dim=-2)
        return y, attention_weight


class PersonalizedAttentivePooling(nn.Module):
    def __init__(self, value_emb_dim, attention_hidden_dim, dropout_rate=0.2):
        super().__init__()
        self.dropouts = nn.Dropout(dropout_rate)
        # build attention network
        self.vector_att = nn.Sequential(nn.Linear(value_emb_dim, attention_hidden_dim), nn.Tanh())

    def forward(self, vec_input, query_input):
        vectors = self.dropouts(vec_input)
        vec_att = self.vector_att(vectors)
        vec_att2 = torch.softmax(torch.bmm(vec_att, query_input.unsqueeze(dim=-1)).squeeze(-1), dim=-1)
        y = torch.bmm(vec_att2.unsqueeze(1), vectors).squeeze(1)
        return y, vec_att2


class MultiHeadedAttention(nn.Module):
    """
    MultiheadedAttention:

    http://nlp.seas.harvard.edu/2018/04/03/attention.html#attention
    """

    def __init__(self, h, d_k, word_dim, dropout=0):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        # We assume d_v always equals d_k
        self.d_k = d_k
        self.h = h
        d_model = h * d_k
        import copy
        self.linears = nn.ModuleList([copy.deepcopy(nn.Linear(word_dim, d_model)) for _ in range(3)])
        self.final = nn.Linear(d_model, d_model)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    @staticmethod
    def attention(query, key, value, mask=None, dropout=None):
        """Compute 'Scaled Dot Product Attention'"""
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = torch.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [liner(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for liner, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.final(x), self.attn


class Conv1D(nn.Module):
    def __init__(self, in_channels: int, kernel_num: int, window_size: int, cnn_method: str = "naive"):
        super(Conv1D, self).__init__()
        assert cnn_method in ['naive', 'group3', 'group5']
        self.cnn_method = cnn_method
        self.in_channels = in_channels
        if self.cnn_method == 'naive':
            self.conv = nn.Conv1d(in_channels=in_channels, out_channels=kernel_num, kernel_size=window_size,
                                  padding=(window_size - 1) // 2)
        elif self.cnn_method == 'group3':
            assert kernel_num % 3 == 0
            self.conv1 = nn.Conv1d(in_channels=self.in_channels, out_channels=kernel_num // 3, kernel_size=1, padding=0)
            self.conv2 = nn.Conv1d(in_channels=self.in_channels, out_channels=kernel_num // 3, kernel_size=3, padding=1)
            self.conv3 = nn.Conv1d(in_channels=self.in_channels, out_channels=kernel_num // 3, kernel_size=5, padding=2)
        else:
            assert kernel_num % 5 == 0
            self.conv1 = nn.Conv1d(in_channels=self.in_channels, out_channels=kernel_num // 5, kernel_size=1, padding=0)
            self.conv2 = nn.Conv1d(in_channels=self.in_channels, out_channels=kernel_num // 5, kernel_size=2, padding=0)
            self.conv3 = nn.Conv1d(in_channels=self.in_channels, out_channels=kernel_num // 5, kernel_size=3, padding=1)
            self.conv4 = nn.Conv1d(in_channels=self.in_channels, out_channels=kernel_num // 5, kernel_size=4, padding=1)
            self.conv5 = nn.Conv1d(in_channels=self.in_channels, out_channels=kernel_num // 5, kernel_size=5, padding=2)

    # Input
    # feature : [batch_size, feature_dim, length]
    # Output
    # out     : [batch_size, kernel_num, length]
    def forward(self, feature):
        if self.cnn_method == 'naive':
            return torch.relu(self.conv(feature))  # [batch_size, kernel_num, length]
        elif self.cnn_method == 'group3':
            return torch.relu(torch.cat([self.conv1(feature), self.conv2(feature), self.conv3(feature)], dim=1))
        else:
            padding_zeros = torch.zeros([feature.size(0), self.in_channels, 1], device=self.device)
            return torch.relu(torch.cat([self.conv1(feature),
                                         self.conv2(torch.cat([feature, padding_zeros], dim=1)),
                                         self.conv3(feature),
                                         self.conv4(torch.cat([feature, padding_zeros], dim=1)),
                                         self.conv5(feature)], dim=1))
