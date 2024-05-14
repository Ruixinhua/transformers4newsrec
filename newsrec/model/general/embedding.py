# -*- coding: utf-8 -*-
# @Author        : Rui
# @Time          : 2024/4/14 16:49
# @Function      : Define the embedding layers for news data
import torch
from torch import nn

from newsrec.utils import load_glove_embedding_matrix, load_tokenizer, load_feature_mapper, load_user_history_mapper


class WordEmbedding(nn.Module):
    def __init__(self, **kwargs):
        super(WordEmbedding, self).__init__()
        self.embedding_type = kwargs.get("embedding_type", "glove")
        if self.embedding_type == "glove":
            glove_embedding = load_glove_embedding_matrix(**kwargs)
            self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(glove_embedding), freeze=False)
            self.embed_dim = glove_embedding.shape[1]
        elif self.embedding_type == "init":
            tokenizer = load_tokenizer(**kwargs)
            self.vocab = tokenizer.get_vocab()
            self.embed_dim = kwargs.get("embed_dim", 300)
            self.embedding = nn.Embedding(len(self.vocab), self.embed_dim, padding_idx=0)
        elif self.embedding_type == "plm":
            # load weight and model from pretrained model
            from transformers import AutoConfig, AutoModel
            self.output_hidden_states = kwargs.get("output_hidden_states", True)
            self.return_attention = kwargs.get("output_attentions", True)
            self.n_layers = kwargs.get("n_layers", 1)
            self.num_classes = kwargs.get("num_classes", 15)
            self.embed_config = AutoConfig.from_pretrained(self.embedding_type, num_labels=self.num_classes,
                                                           output_hidden_states=self.output_hidden_states,
                                                           output_attentions=self.return_attention)
            add_weight = kwargs.get("add_weight", False)
            try:
                self.embed_config.__dict__.update({"add_weight": add_weight, "n_layers": self.n_layers})
            except AttributeError:
                self.embed_config.__dict__.update({"add_weight": add_weight, "num_hidden_layers": self.n_layers})
            if self.embedding_type == "allenai/longformer-base-4096":
                self.embed_config.attention_window = self.embed_config.attention_window[:self.n_layers]
            embedding = AutoModel.from_pretrained(self.embedding_type, config=self.embed_config)
            self.embedding = kwargs.get("bert")(self.embed_config) if "bert" in kwargs else embedding
            if hasattr(self.embed_config, "dim"):  # for roberta like language model
                self.embed_dim = self.embed_config.dim
            elif hasattr(self.embed_config, "hidden_size"):  # for bert like language model
                self.embed_dim = self.embed_config.hidden_size
            else:
                raise ValueError("Unsure the embedding dimension, please check the config of the model")
        else:
            raise ValueError("Unknown embedding type")

    def forward(self, news_tokens, news_mask=None, **kwargs):
        """
        Get the word embeddings of news tokens
        :param news_tokens: size = (batch_size, max_news_length)
        :param news_mask: size = (batch_size, max_news_length)
        :param kwargs:
        :return: word embeddings of news tokens
        """
        if self.embedding_type in ["glove", "init"]:
            word_embed = self.embedding(news_tokens)
        else:  # for bert like language model
            input_embeds = kwargs.get("embedding", None)
            word_embed = self.embedding(news_tokens, news_mask, inputs_embeds=input_embeds)[0]
        return word_embed


class FeatureEmbedding(nn.Module):
    def __init__(self, **kwargs):
        super(FeatureEmbedding, self).__init__()
        self.title_len = kwargs.get("title_len", 30)
        self.abstract_len = kwargs.get("abstract_len", 30)
        self.body_len = kwargs.get("body_len", 100)
        self.embed_dim = self.title_len + self.abstract_len + self.body_len + 2
        feature_mapper = load_feature_mapper(**kwargs)
        self.entity_feature = kwargs.get("entity_feature")
        if self.entity_feature:
            self.entity_dict = feature_mapper.entity_dict
        self.tokenizer = feature_mapper.tokenizer
        self.category_mapper = feature_mapper.category_mapper
        self.subvert_mapper = feature_mapper.subvert_mapper
        self.length = len(feature_mapper.feature_matrix)
        self.embedding = nn.Embedding.from_pretrained(torch.LongTensor(feature_mapper.feature_matrix), freeze=True)

    def forward(self, nid):
        """
        Get the feature embeddings of input news id
        :param nid: size = (batch_size, history_len+candidate_len)
        :return: feature embeddings
        """
        return self.embedding(nid)

    def select_feature(self, f_vec):
        """
        Select the features of news
        :param f_vec: feature vectors, shape=(B, X, F)
        :return: feature vectors of news
        """
        text_len = self.title_len + self.abstract_len + self.body_len
        feature_dict = {
            "title": f_vec[..., :self.title_len], "body": f_vec[..., self.title_len+self.abstract_len:text_len],
            "abstract": f_vec[..., self.title_len:self.title_len+self.abstract_len],
            "category": f_vec[..., text_len], "subvert": f_vec[..., text_len+1]
        }
        if self.entity_feature:
            feature_dict["entity"] = f_vec[..., text_len+2:]
        return feature_dict

    def __len__(self):
        return self.length


class UserHistoryEmbedding(nn.Module):
    def __init__(self, **kwargs):
        super(UserHistoryEmbedding, self).__init__()
        self.user_history_mapper = load_user_history_mapper(**kwargs)
        self.length = len(self.user_history_mapper)
        self.embedding = nn.Embedding.from_pretrained(torch.LongTensor(self.user_history_mapper), freeze=True)

    def forward(self, uid):
        return self.embedding(uid)

    def __len__(self):
        return self.length


class FrozenEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, embedding_matrix=None, **kwargs):
        super(FrozenEmbedding, self).__init__()
        if embedding_matrix:
            self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix), freeze=True)
        else:
            self.length = num_embeddings
            self.embedding = nn.Embedding(num_embeddings, embedding_dim, **kwargs)
            # Freeze the parameters of the embedding layer
            for param in self.embedding.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.embedding(x)

    def __len__(self):
        return self.length

