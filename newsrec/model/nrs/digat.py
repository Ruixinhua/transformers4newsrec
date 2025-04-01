# -*- coding: utf-8 -*-
# @Author        : Ruoyang
# @Time          : 2024/9/11 14:49
# @Function      : Implement the model of DIGAT
import torch
from .base import BaseNRS


class DIGATRSModel(BaseNRS):
    """
    DIGAT: Modeling News Recommendation with Dual-Graph Interaction
    Paper of DIGAT: https://aclanthology.org/2022.findings-emnlp.491.pdf
    Source library: https://github.com/Veason-silverbullet/DIGAT
    """
    def __init__(self, **kwargs):
        super(DIGATRSModel, self).__init__(**kwargs)
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
        return super().news_encoder(input_feat)  # placeholder implementation

    def user_encoder(self, input_feat):
        """

        :param input_feat:
        :return:
        """
        local_history_news = input_feat["history_news"]  # shape = (B, H, D)
        candidate_news_vector = self.get_mapping_vector(input_feat["news_vector"], input_feat["candidate_mapping"])
        ...
        return super().user_encoder(input_feat)  # placeholder implementation

    def predict(self, candidate_news_vector, user_vector, **kwargs):
        """

        :param candidate_news_vector:
        :param user_vector:
        :return:
        """
        ...
        return super().predict(candidate_news_vector, user_vector)  # placeholder implementation
