# -*- coding: utf-8 -*-
# @Author        : Rui
# @Time          : 2024/4/14 18:57
# @Function      : Basic model for news recommendation system
import torch
import torch.nn as nn
import newsrec.utils.loss_utils as module_loss

from torch.utils.data import DataLoader
from newsrec.model.base_model import BaseModel
from newsrec.model.general import WordEmbedding, FeatureEmbedding, FrozenEmbedding, UserHistoryEmbedding
from newsrec.model.general import ClickPredictor, AttLayer
from newsrec.utils import reshape_tensor, load_tokenizer, load_entity_embedding_matrix


class BaseNRS(BaseModel):
    def __init__(self, **kwargs):
        """
        Base class for News Recommendation System
        :param kwargs:
        """
        super(BaseNRS, self).__init__()
        self.word_embedding = WordEmbedding(**kwargs)
        self.feature_embedding = FeatureEmbedding(**kwargs)
        if not hasattr(self, "embedding_dim"):
            # embedding_dim is the dimension of the final news embedding and user embedding for recommendation
            self.embedding_dim = self.word_embedding.embed_dim
        self.news_batch_size = kwargs.get("news_batch_size", 1024)
        self.user_batch_size = kwargs.get("user_batch_size", 128)
        self.user_history = UserHistoryEmbedding(**kwargs)
        self.nid_embedding = FrozenEmbedding(len(self.feature_embedding), self.embedding_dim)  # news_num, embed_dim
        self.uid_embedding = FrozenEmbedding(len(self.user_history), self.embedding_dim)  # user_num, embed_dim
        self.load_embedding = False  # set to false when training and at the beginning of the evaluation
        self.fast_evaluation = kwargs.get("fast_evaluation", False)
        self.criterion = getattr(module_loss, kwargs.get("loss", "categorical_loss"))
        self.pad_token_id = int(load_tokenizer(**kwargs).pad_token_id)
        # news feature can be used: title, abstract, body, category, subvert
        self.text_feature = kwargs.get("text_feature", ["title"])
        self.cat_feature = kwargs.get("cat_feature", [])
        self.entity_feature = kwargs.get("entity_feature")
        self.use_layernorm = kwargs.get("use_layernorm", False)  # whether to use layernorm in attention layer
        self.use_flash_att = kwargs.get("use_flash_att", False)
        self.use_news_mask = kwargs.get("use_news_mask", False)  # whether to use news mask
        self.use_user_mask = kwargs.get("use_user_mask", False)  # whether to use user mask
        if isinstance(self.entity_feature, str):
            self.entity_feature = self.entity_feature.split(",")  # split entity feature by ","
        if self.cat_feature and len(self.cat_feature):
            self.category_dim = kwargs.get("category_dim", 100)
            if "category" in self.cat_feature:
                cat_len = len(self.feature_embedding.category_mapper) + 1
                self.category_embedding = nn.Embedding(cat_len, self.category_dim)
            if "subvert" in self.cat_feature:
                sub_len = len(self.feature_embedding.subvert_mapper) + 1
                self.subvert_embedding = nn.Embedding(sub_len, self.category_dim)
        if self.entity_feature:
            entity_embed_matrix = load_entity_embedding_matrix(self.feature_embedding.entity_dict, **kwargs)
            self.entity_dim = entity_embed_matrix.shape[1]
            self.entity_embedding = nn.Embedding.from_pretrained(
                torch.FloatTensor(entity_embed_matrix), freeze=False, padding_idx=0
            )
        if self.use_layernorm:
            # self.multi_layer_norm = nn.LayerNorm(self.embedding_dim)
            self.att_layer_norm = nn.LayerNorm(self.embedding_dim)
        self.click_predictor = ClickPredictor(**kwargs)
        # self.uid_feature = kwargs.get("uid_feature")
        self.attention_hidden_dim = kwargs.get("attention_hidden_dim", 200)
        self.dropout_we = nn.Dropout(kwargs.get("dropout_we", 0.2))  # dropout for word embedding
        self.dropout_ne = nn.Dropout(kwargs.get("dropout_ne", 0.2))  # dropout for news encoder layer
        self.dropout_ce = nn.Dropout(kwargs.get("dropout_ce", 0.2))  # dropout for category encoder layer
        self.news_layer = AttLayer(self.embedding_dim, self.attention_hidden_dim)
        self.user_layer = AttLayer(self.embedding_dim, self.attention_hidden_dim)

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
        # news_tokens: shape = (B, H+C, F), F is the sum of feature used in news
        if len(news_tokens.shape) == 3:
            news_tokens = reshape_tensor(news_tokens)  # out: (B * (H+C), F)
        news_mask = torch.where(news_tokens == self.pad_token_id, 0, 1).to(news_tokens.device)  # shape = (B*(H+C), F)
        word_vector = self.dropout_we(self.word_embedding(news_tokens, news_mask))  # shape = (B*(H+C), F, E)
        return word_vector, news_mask

    def news_encoder(self, input_feat):
        """
        Encode news using text feature encoder with news attention layer
        :param input_feat: history_nid, candidate_nid; shape = (B, H), (B, C);
        :return: news vector, shape = (B*(H+C), E); news weight, shape = (B*(H+C), F)
        """
        word_vector, news_mask = self.text_feature_encoder(input_feat)  # shape = (B*(H+C), F, E)
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

    def compute_loss(self, prediction, label):
        loss = self.criterion(prediction, label)
        return loss

    @staticmethod
    def get_mapping_index(unique_nid, *batch_nid):
        nid_mapper = torch.full((unique_nid.max().item() + 1,), -1, dtype=torch.long)
        nid_mapper[unique_nid] = torch.arange(unique_nid.size(0))
        nid_mapper = nid_mapper.to(batch_nid[0].device)
        return (nid_mapper[nid] for nid in batch_nid)

    @staticmethod
    def get_mapping_vector(source_vectors, mapping_idx):
        mask = (mapping_idx == -1)
        # add extra dimension for mask if the dimension of mask is less than source_vectors
        if len(mask.shape) <= len(source_vectors.shape):
            mask = mask.unsqueeze(-1)
        return source_vectors[mapping_idx, ...].masked_fill(mask, 0)

    def build_input_feat(self, input_feat):
        history_nid, history_mask = input_feat["history_nid"], input_feat["history_mask"]
        candidate_nid, candidate_mask = input_feat["candidate_nid"], input_feat["candidate_mask"]
        history_selected = torch.masked_select(history_nid, history_mask)  # select history based on mask
        candidate_selected = torch.masked_select(candidate_nid, candidate_mask)  # select candidate based on mask
        # get unique news id in batch of history and candidate
        input_feat["nid"] = torch.unique(torch.cat((history_selected, candidate_selected), dim=0))
        input_feat["history_mapping"], input_feat["candidate_mapping"] = self.get_mapping_index(
            input_feat["nid"], history_nid, candidate_nid
        )  # -1 means zero padding; history_mapping: B, H; candidate_mapping: B, C
        return input_feat

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
        if self.training:  # keep load embedding to false when training
            self.load_embedding = False
        input_feat = {"uid": uid, "history_nid": history_nid, "candidate_nid": candidate_nid,
                      "history_mask": history_mask, "candidate_mask": candidate_mask}
        input_feat = self.build_input_feat(input_feat)
        if not self.training and not self.load_embedding and self.fast_evaluation:
            """run model to generate embedding caches for evaluation"""
            self.load_embedding = True
            torch.cuda.empty_cache()  # empty training cache
            news_indices = torch.tensor(range(len(self.nid_embedding))).to(uid.device)
            news_loader = DataLoader(news_indices, batch_size=self.news_batch_size, shuffle=False)
            news_embeddings = torch.zeros(len(self.nid_embedding), self.embedding_dim).to(uid.device)
            for nid in news_loader:
                news_embeddings[nid] = self.news_encoder({"nid": nid})["news_vector"]
            self.nid_embedding.embedding = nn.Embedding.from_pretrained(news_embeddings, freeze=True).to(uid.device)
            user_indices = torch.tensor(range(len(self.uid_embedding))).to(uid.device)
            user_loader = DataLoader(user_indices, batch_size=self.user_batch_size, shuffle=False)
            user_embedding = torch.zeros(len(self.uid_embedding), self.embedding_dim).to(uid.device)
            for u in user_loader:
                h_nid = self.user_history(u)
                history_news = self.nid_embedding(h_nid)
                user_input = {"uid": u, "history_mask": h_nid != 0, "history_news": history_news}
                user_embedding[u] = self.user_encoder(user_input)["user_vector"]
            self.uid_embedding.embedding = nn.Embedding.from_pretrained(user_embedding, freeze=True).to(uid.device)
            torch.cuda.empty_cache()  # empty cuda cache
        if self.load_embedding:
            candidate_news_vector = self.nid_embedding(candidate_nid)
            user_vector = self.uid_embedding(uid)
        else:
            news_feature = self.news_encoder(input_feat)  # shape = (B*(H+C), E)
            news_vector = news_feature["news_vector"]
            # fetch history news vector from all news vectors
            input_feat["history_news"] = self.get_mapping_vector(news_vector, input_feat["history_mapping"])
            input_feat.update(news_feature)
            # run user encoder
            output_dict = self.user_encoder(input_feat)
            user_vector = output_dict["user_vector"]
            # candidate_news_vector = news_vector[:, history_nid.size(1):, :]
            # fetch candidate news vector from all news vectors
            candidate_vector = self.get_mapping_vector(news_vector, input_feat["candidate_mapping"])
            candidate_news_vector = output_dict.get("candidate_news_vector", candidate_vector)
        prediction = self.predict(candidate_news_vector, user_vector)
        loss = self.compute_loss(prediction, kwargs.get("label"))
        model_output = {"loss": loss, "prediction": prediction}
        return model_output
