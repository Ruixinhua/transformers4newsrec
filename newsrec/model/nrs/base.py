# -*- coding: utf-8 -*-
# @Author        : Rui
# @Time          : 2024/4/14 18:57
# @Function      : Basic model for news recommendation system
import torch
import torch.nn as nn

from newsrec.model.base_model import BaseModel
from newsrec.model.general import WordEmbedding, FeatureEmbedding, ClickPredictor, AttLayer
from newsrec.utils import reshape_tensor, load_tokenizer
import newsrec.utils.loss_utils as module_loss


class BaseNRS(BaseModel):
    def __init__(self, **kwargs):
        """
        Base class for News Recommendation System
        :param kwargs:
        """
        super(BaseNRS, self).__init__()
        self.word_embedding = WordEmbedding(**kwargs)
        self.feature_embedding = FeatureEmbedding(**kwargs)
        self.click_predictor = ClickPredictor(**kwargs)
        self.criterion = getattr(module_loss, kwargs.get("loss", "categorical_loss"))
        self.pad_token_id = int(load_tokenizer(**kwargs).pad_token_id)
        # news feature can be used: title, abstract, body, category, subvert
        self.text_feature = kwargs.get("text_feature", ["title"])
        self.cat_feature = kwargs.get("cat_feature")
        # self.uid_feature = kwargs.get("uid_feature")
        self.attention_hidden_dim = kwargs.get("attention_hidden_dim", 200)
        self.dropout_we = nn.Dropout(kwargs.get("dropout_we", 0.2))  # dropout for word embedding
        self.embedding_dim = self.word_embedding.embed_dim
        self.news_layer = AttLayer(self.embedding_dim, self.attention_hidden_dim)
        self.user_layer = AttLayer(self.embedding_dim, self.attention_hidden_dim)

    def text_feature_encoder(self, input_feat):
        """
        Encode text feature of news using word embedding layer
        :param input_feat: history_nid, candidate_nid; shape = (B, H), (B, C);
        :return: word vector of news, shape = (B, H+C, E), E is word embedding dimension
        """
        # concat history news and candidate news
        news_all = torch.cat((input_feat["history_nid"], input_feat["candidate_nid"]), dim=1)
        news_all_feature = self.feature_embedding(news_all)  # shape = (B, H+C, F)
        news_feature_dict = self.feature_embedding.select_feature(news_all_feature)
        news_tokens = torch.cat([news_feature_dict[feature] for feature in self.text_feature], dim=2)
        # news_tokens: shape = (B, H+C, F), F is the sum of feature used in news
        news_tokens = reshape_tensor(news_tokens)  # out: (B * (H+C), F)
        news_mask = torch.where(news_tokens == self.pad_token_id, 0, 1).to(news_tokens.device)  # shape = (B*(H+C), F)
        word_vector = self.dropout_we(self.word_embedding(news_tokens, news_mask))  # shape = (B*(H+C), F, E)
        return word_vector

    def news_encoder(self, input_feat):
        """
        Encode news using text feature encoder with news attention layer
        :param input_feat: history_nid, candidate_nid; shape = (B, H), (B, C);
        :return: news vector, shape = (B*(H+C), E); news weight, shape = (B*(H+C), F)
        """
        word_vector = self.text_feature_encoder(input_feat)  # shape = (B*(H+C), F, E)
        output = self.news_layer(word_vector)  # shape = (B*(H+C), E)
        return {"news_vector": output[0], "news_weight": output[1]}

    def user_encoder(self, input_feat):
        """
        Encode user using history news with user attention layer
        :param input_feat: history_news, shape = (B, H, E)
        :return: user_vector, shape = (B, E); user_weight, shape = (B, H)
        """
        y = self.user_layer(input_feat["history_news"])  # shape = (B, H, E)
        return {"user_vector": y[0], "user_weight": y[1]}

    def predict(self, candidate_news_vector, user_vector):
        """
        prediction logic: use MLP or Dot-product for prediction
        :param candidate_news_vector: shape = (B, C, D)
        :param user_vector: shape = (B, D)
        :return: softmax possibility of click candidate news
        """
        return self.click_predictor(candidate_news_vector, user_vector)

    def compute_loss(self, input_feat, label):
        loss = self.criterion(input_feat["prediction"], label)
        return loss

    def forward(self, uid, history_nid, candidate_nid, history_mask, candidate_mask, **kwargs):
        """
        B is batch size, H is maximum history size, C is candidate size, F is feature size
        :param uid: user id, shape = (B)
        :param history_nid: history news id, shape = (B, H)
        :param candidate_nid: candidate news id, shape = (B, C)
        :param history_mask: history news mask, shape = (B, H)
        :param candidate_mask: candidate news mask, shape = (B, C)
        :return: click possibility of candidate news, shape = (B, C)
        """
        input_feat = {"uid": uid, "history_nid": history_nid, "candidate_nid": candidate_nid,
                      "history_mask": history_mask, "candidate_mask": candidate_mask}
        output_dict = self.news_encoder(input_feat)  # shape = (B*(H+C), E)
        news_vector = output_dict["news_vector"]
        # reshape news vector to history news vector and candidate news vector -> (B, H+C, E)
        news_vector = reshape_tensor(news_vector, (uid.size(0), -1, news_vector.size(-1)))
        input_feat["history_news"] = news_vector[:, :history_nid.size(1), :]
        output_dict = self.user_encoder(input_feat)
        user_vector = output_dict["user_vector"]
        candidate_news_vector = news_vector[:, history_nid.size(1):, :]
        input_feat["prediction"] = self.predict(candidate_news_vector, user_vector)
        loss = self.compute_loss(input_feat, kwargs.get("label"))
        model_output = {"loss": loss, "prediction": input_feat["prediction"]}
        return model_output
