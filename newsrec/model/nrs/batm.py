import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence
from newsrec.model.general import BiAttentionLayer, AttLayer
from .base import BaseNRS


class BATMRSModel(BaseNRS):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bi_attention_layer = BiAttentionLayer(**kwargs)
        self.news_encoder_name = kwargs.get("news_encoder_name", "base")
        self.user_encoder_name = kwargs.get("user_encoder_name", "bi_attention")
        self.topic_num, self.topic_dim = kwargs.get("topic_num", 50), kwargs.get("topic_dim", 20)
        topic_dim = self.topic_num * self.topic_dim
        # the structure of basic model
        if self.user_encoder_name == "gru":
            self.user_encode_layer = nn.GRU(self.embedding_dim, self.embedding_dim, batch_first=True)
        elif self.user_encoder_name == "batm":
            self.user_encode_layer = nn.Sequential(nn.Linear(self.embedding_dim, topic_dim), nn.Tanh(),
                                                   nn.Linear(topic_dim, self.topic_num))
            self.user_final = nn.Linear(self.embedding_dim, self.embedding_dim)
        if self.news_encoder_name == "multi_view":
            self.topic_att = AttLayer(self.embedding_dim * 2, self.attention_hidden_dim)
            self.topic_affine = nn.Linear(self.embedding_dim * 2, self.embedding_dim)

    def extract_topic(self, input_feat):
        word_vector, news_mask = self.text_feature_encoder(input_feat)  # shape = (B*(H+C), F, E)
        return self.bi_attention_layer(word_vector, news_mask)

    def news_encoder(self, input_feat):
        """input_feat: Size is [N * H, S]"""
        if self.news_encoder_name == "multi_view":
            title = self.dropout_ne(self.embedding_layer(news=input_feat["title"], news_mask=input_feat["title_mask"]))
            body = self.dropout_ne(self.embedding_layer(news=input_feat["body"], news_mask=input_feat["body_mask"]))
            title_topics = self.topic_layer(news_embeddings=title, news_mask=input_feat["title_mask"])
            topic_dict = self.topic_layer(news_embeddings=body, news_mask=input_feat["body_mask"])
            topic_weight = torch.cat([title_topics["topic_weight"], topic_dict["topic_weight"]], dim=-1)
            all_vec = torch.cat([title_topics["topic_vector"], topic_dict["topic_vector"]], dim=-1)
            topic_vector, news_weight = self.topic_att(all_vec)
            news_vector = torch.relu(self.topic_affine(topic_vector))
        else:
            topic_dict = self.extract_topic(input_feat)
            topic_weight = topic_dict["topic_weight"]
            # add activation function
            news_vector, news_weight = self.news_layer(self.dropout_ne(topic_dict["topic_vector"]))
        out_dict = {"news_vector": news_vector, "news_weight": news_weight.squeeze(-1), "topic_weight": topic_weight}
        out_dict.update(topic_dict)
        return out_dict

    def user_encoder(self, input_feat):
        history_news = input_feat["history_news"]
        if self.user_encoder_name == "gru":
            history_length = torch.sum(input_feat["history_mask"], dim=-1).cpu()
            history_length[history_length == 0] = 1  # avoid zero history recording
            packed_y = pack_padded_sequence(history_news, history_length, batch_first=True, enforce_sorted=False)
            user_vector = self.user_encode_layer(packed_y)[1].squeeze(dim=0)
            user_weight = None
            # y = self.user_encode_layer(history_news)[0]
            # user_vector, user_weight = self.user_layer(y)  # additive attention layer
        elif self.user_encoder_name == "batm":
            user_weight = self.user_encode_layer(history_news).transpose(1, 2)
            # mask = input_feat["news_mask"].expand(self.topic_num, y.size(0), -1).transpose(0, 1) == 0
            # user_weight = torch.softmax(user_weight.masked_fill(mask, 0), dim=-1)  # fill zero entry with zero weight
            user_vec = self.user_final(torch.matmul(user_weight, history_news))
            user_vector, user_weight = self.user_layer(user_vec)  # additive attention layer
        else:
            user_vector, user_weight = self.user_layer(history_news)  # additive attention layer
        return {"user_vector": user_vector, "user_weight": user_weight}
