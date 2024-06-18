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

    def __init__(self, emb_dim, attention_hidden_dim):
        super().__init__()
        # build attention network
        self.attention = nn.Sequential(
            nn.Linear(emb_dim, attention_hidden_dim, bias=True),
            nn.Tanh(),
            nn.Linear(attention_hidden_dim, 1, bias=True),
        )

    def initialize(self):
        nn.init.xavier_uniform_(self.att_fc1.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.zeros_(self.att_fc1.bias)
        nn.init.xavier_uniform_(self.att_fc2.weight)

    def forward(self, x, x_mask=None):
        attention_weight = self.attention(x)
        attention_weight = torch.exp(attention_weight)
        if x_mask is not None:
            attention_weight = attention_weight * x_mask.unsqueeze(dim=-1)
        attention_weight = attention_weight / (torch.sum(attention_weight, dim=1, keepdim=True) + 1e-8)
        y = torch.sum(x * attention_weight, dim=-2)
        return y, attention_weight.squeeze(-1)


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

    def __init__(self, h, d_k, word_dim, dropout=0, use_flash_att=True):
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
        self.use_flash_att = use_flash_att
        self.apply(lambda layer: nn.init.xavier_uniform_(layer.weight) if isinstance(layer, nn.Linear) else None)

    @staticmethod
    def attention(query, key, value, mask=None, dropout=None):
        """Compute 'Scaled Dot Product Attention'"""
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        scores = torch.exp(scores)
        if mask is not None:
            scores = scores * mask

        p_attn = scores / (torch.sum(scores, dim=-1, keepdim=True) + 1e-8)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

    def forward(self, query, key, value, mask=None):
        """

        :param query: batch_size, word_num, word_dim
        :param key: batch_size, word_num, word_dim
        :param value: batch_size, word_num, word_dim
        :param mask: batch_size, word_num
        :return: output: batch_size, word_num, word_dim; weight:
        """
        if mask is not None:
            # Same mask applied to all h heads.
            # mask = mask.unsqueeze(1)
            mask = mask.unsqueeze(dim=1).expand(-1, self.h, -1).unsqueeze(-1)
        nbatches = query.size(0)

        if self.use_flash_att:
            from flash_attn import flash_attn_qkvpacked_func
            qkv = torch.stack([liner(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                               for liner, x in zip(self.linears, (query, key, value))], dim=2)
            x = flash_attn_qkvpacked_func(qkv)
        else:
            # 1) Do all the linear projections in batch from d_model => h x d_k
            query, key, value = [liner(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                                 for liner, x in zip(self.linears, (query, key, value))]
            # 2) Apply attention on all the projected vectors in batch.
            x, self.attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.final(x), self.attn


class BiAttentionLayer(nn.Module):
    def __init__(self, **model_args):
        super(BiAttentionLayer, self).__init__()
        self.topic_layer_name = model_args.get("topic_layer_name", "base_att")
        self.topic_num, self.topic_dim = model_args.get("topic_num", 50), model_args.get("topic_dim", 20)
        # self.topic_dim = model_args.get("topic_dim", self.topic_num * self.topic_dim)
        self.embedding_dim = model_args.get("embedding_dim", 300)
        self.hidden_dim = model_args.get("hidden_dim", 256)
        self.add_topic_evaluation = model_args.get("add_topic_evaluation", False)
        self.act_layer = activation_layer(model_args.get("act_layer", "tanh"))  # default use tanh
        self.final = nn.Linear(self.embedding_dim, self.embedding_dim)
        if self.topic_layer_name == "base_att":
            self.topic_layer = nn.Sequential(
                nn.Linear(self.embedding_dim, self.topic_num * self.topic_dim), self.act_layer,
                nn.Linear(self.topic_num * self.topic_dim, self.topic_num)
            )
        elif self.topic_layer_name == "base_advanced":
            self.topic_layer = nn.Sequential(
                # map to hidden dim
                nn.Linear(self.embedding_dim, self.topic_num * self.hidden_dim), self.act_layer,
                # map to topic dim
                nn.Linear(self.topic_num * self.hidden_dim, self.topic_num * self.topic_dim), self.act_layer,
                # map to topic num
                nn.Linear(self.topic_num * self.topic_dim, self.topic_num), activation_layer("sigmoid")
            )
        elif self.topic_layer_name == "base_topic_vector":
            self.topic_layer = nn.Linear(self.embedding_dim, self.topic_num, bias=False)
        elif self.topic_layer_name == "variational_topic":
            self.topic_layer = nn.Sequential(
                nn.Linear(self.embedding_dim, self.topic_num * self.hidden_dim, bias=True), self.act_layer,
                nn.Linear(self.topic_num * self.hidden_dim, self.topic_num, bias=True)
            )
            self.logsigma_q_theta = nn.Sequential(
                nn.Linear(self.embedding_dim, self.topic_num * self.hidden_dim, bias=True), self.act_layer,
                nn.Linear(self.topic_num * self.hidden_dim, self.topic_num, bias=True)
            )
        else:
            raise ValueError("Specify correct variant name!")

    def forward(self, news_embeddings, news_mask):
        """
        Topic forward pass, return topic vector and topic weights
        """
        out_dict = {}
        if self.topic_layer_name == "variational_topic":
            scores = self.topic_layer(news_embeddings)
            log_q_theta = self.logsigma_q_theta(news_embeddings)
            out_dict["kl_divergence"] = -0.5 * torch.sum(1+log_q_theta-scores.pow(2)-log_q_theta.exp(), dim=-1).mean()
            if self.training:  # reparameterization topic weight in training
                std = torch.exp(0.5 * log_q_theta)
                eps = torch.randn_like(std)
                scores = eps.mul_(std).add_(scores)
            scores = scores.transpose(1, 2)
        elif self.topic_layer_name == "base_topic_vector":
            scores = (news_embeddings @ self.topic_layer.weight.transpose(0, 1)).transpose(1, 2)  # (N, H, S)
        else:
            scores = self.topic_layer(news_embeddings).transpose(1, 2)  # (N, H, S)
        weights = torch.exp(scores) * news_mask.unsqueeze(dim=1)
        topic_weight = weights / (torch.sum(weights, dim=-1, keepdim=True) + 1e-8)
        topic_vector = self.final(torch.matmul(topic_weight, news_embeddings))  # (N, H, E)
        out_dict.update({"topic_vector": topic_vector, "topic_weight": topic_weight})
        return out_dict


class MultiFeatureAttentionFusion(nn.Module):
    def __init__(self, feature_dims, fusion_dim):
        super(MultiFeatureAttentionFusion, self).__init__()
        self.feature_transforms = nn.ModuleList([
            nn.Linear(dim, fusion_dim) for dim in feature_dims
        ])
        self.fusion_dim = fusion_dim
        self.query = nn.Linear(fusion_dim, fusion_dim)
        self.key = nn.Linear(fusion_dim, fusion_dim)
        self.value = nn.Linear(fusion_dim, fusion_dim)

    def forward(self, features):
        # Transform each feature to the fusion dimension
        transformed_features = [trans(feat) for trans, feat in zip(self.feature_transforms, features)]

        # Concatenate all features
        concat_features = torch.cat(transformed_features, dim=1)

        # Apply attention
        q = self.query(concat_features)
        k = self.key(concat_features)
        v = self.value(concat_features)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.fusion_dim ** 0.5)
        attention_weights = torch.softmax(scores, dim=-1)
        fused_feature = torch.matmul(attention_weights, v)

        return fused_feature


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


def activation_layer(act_name):
    """
    Construct activation layers
    :param act_name: str or nn.Module, name of activation function
    :return: act_layer: activation layer
    """
    act_layer = None
    if isinstance(act_name, str):
        if act_name.lower() == "sigmoid":
            act_layer = nn.Sigmoid()
        elif act_name.lower() == "relu":
            act_layer = nn.ReLU(inplace=True)
        elif act_name.lower() == "prelu":
            act_layer = nn.PReLU()
        elif act_name.lower() == "tanh":
            act_layer = nn.Tanh()
    elif issubclass(act_name, nn.Module):
        act_layer = act_name()
    else:
        raise NotImplementedError(f"Activation layer {act_name} is not implemented")
    return act_layer
