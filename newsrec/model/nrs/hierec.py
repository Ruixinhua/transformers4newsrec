# -*- coding: utf-8 -*-
# @Author        : Xiaojun
# @Time          : 2024/9/11 14:49
# @Function      : Implement the model of HieRec
import torch
from .base import BaseNRS


class HieRecRSModel(BaseNRS):
    """
    HieRec: Hierarchical User Interest Modeling for Personalized News Recommendation
    Paper of HieRec: https://arxiv.org/pdf/2106.04408
    Source library: https://github.com/taoqi98/HieRec
    """
    def __init__(self, **kwargs):
        super(HieRecRSModel, self).__init__(**kwargs)
        """Implement the architecture of DIGAT"""

    def build_input_feat(self, input_feat):
        """
        Build input features for model
        :param input_feat: history_nid, history_mask, candidate_nid, candidate_mask
        :return: input_feat
        """
        """Select history and candidate news id based on mask, remove padding news id"""
        history_nid, history_mask = input_feat["history_nid"], input_feat["history_mask"]
        candidate_nid, candidate_mask = input_feat["candidate_nid"], input_feat["candidate_mask"]
        history_selected = torch.masked_select(history_nid, history_mask)  # select history based on mask
        candidate_selected = torch.masked_select(candidate_nid, candidate_mask)  # select candidate based on mask
        """Define history and candidate mapping index"""
        # get unique news id in batch of history and candidate
        input_feat["nid"] = torch.unique(torch.cat((history_selected, candidate_selected), dim=0))
        # get mapping index for history and candidate
        input_feat["history_mapping"], input_feat["candidate_mapping"] = self.get_mapping_index(
            input_feat["nid"], history_nid, candidate_nid
        )  # -1 means zero padding; history_mapping: B, H; candidate_mapping: B, C
        return input_feat

    def news_encoder(self, input_feat):
        """

        :param input_feat:
        :return:
        """
        ...
        return self.news_encoder(input_feat)  # placeholder implementation

    def user_encoder(self, input_feat):
        """

        :param input_feat:
        :return:
        """
        local_history_news = input_feat["history_news"]  # shape = (B, H, D)
        candidate_news_vector = self.get_mapping_vector(input_feat["news_vector"], input_feat["candidate_mapping"])
        ...
        return self.user_encoder(input_feat)  # placeholder implementation

    def predict(self, candidate_news_vector, user_vector):
        """

        :param candidate_news_vector:
        :param user_vector:
        :return:
        """
        ...
        return self.predict(candidate_news_vector, user_vector)  # placeholder implementation
