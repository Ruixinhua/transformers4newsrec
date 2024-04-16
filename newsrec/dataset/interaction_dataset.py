# -*- coding: utf-8 -*-
# @Author        : Rui
# @Time          : 2024/4/11 3:30
# @Function      : Define the Interaction Dataset for training, validation and testing
import torch
import numpy as np
from torch.utils.data.dataset import Dataset

from newsrec.utils import load_dataset_from_csv, news_sampling


class UserInteractionDataset(Dataset):
    def __init__(self, **kwargs):
        self.subset_name = kwargs.get("subset_name", "small")
        self.split = kwargs.get("split", "train")
        self.neg_sample_num = kwargs.get("neg_sample_num", 4)
        max_history_size = kwargs.get("max_history_size", 50)
        self.user_interaction = load_dataset_from_csv(f"user_interaction_{self.subset_name}")
        self.impression = load_dataset_from_csv(f"{self.split}_{self.subset_name}")
        self.uid = np.array(self.impression["uid"], dtype=np.int32)
        if self.split == "train":
            self.positive = np.array(self.impression["positive"], dtype=np.int32)
            self.negative = list(self.impression["negative"])
        else:
            self.interaction = list(self.impression["impressions"])
        self.user_history_mapper = np.zeros((len(self.user_interaction) + 1, max_history_size), dtype=np.int32)
        # fetch uid and history to two lists
        history_nid, history_uid = list(self.user_interaction["history"]), list(self.user_interaction["uid"])
        for index in range(len(self.user_interaction)):
            history = history_nid[index]
            history = history.split() if history else [0]
            history = history[-max_history_size:] + [0] * (max_history_size - len(history))
            self.user_history_mapper[history_uid[index]] = np.asarray(history, dtype=np.int32)

    def __getitem__(self, index):
        uid = self.uid[index]
        history_nid = self.user_history_mapper[uid]
        history_mask = np.asarray(history_nid != 0, dtype=np.int8)
        if self.split == "train":
            pos = self.positive[index]
            neg = self.negative[index].split()
            candidate_nid = np.asarray(
                [pos] + news_sampling(neg, self.neg_sample_num), dtype=np.int32
            )
            label = [1] + [0] * self.neg_sample_num
        else:
            interaction = self.interaction[index].split()
            candidate_nid = np.asarray([i.split("-")[0] for i in interaction], dtype=np.int32)
            label = [int(i.split("-")[1]) if "-" in i else -1 for i in interaction]  # -1 for unlabeled data
        candidate_mask = np.asarray(candidate_nid != 0, dtype=np.int8)
        input_feat = {
            "uid": torch.tensor(uid, dtype=torch.int32),
            "history_nid": torch.tensor(history_nid, dtype=torch.int32),
            "history_mask": torch.tensor(history_mask, dtype=torch.int8),
            "candidate_nid": torch.tensor(candidate_nid, dtype=torch.int32),
            "candidate_mask": torch.tensor(candidate_mask, dtype=torch.int8),
            "label": torch.tensor(label, dtype=torch.int8),
        }
        return input_feat

    def __len__(self):
        return len(self.impression)
