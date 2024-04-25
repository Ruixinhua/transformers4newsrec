# -*- coding: utf-8 -*-
# @Author        : Rui
# @Time          : 2024/4/11 3:30
# @Function      : Define the Interaction Dataset for training, validation and testing
import torch
import numpy as np
from torch.utils.data.dataset import Dataset

from newsrec.utils import load_dataset_from_csv, news_sampling, load_user_history_mapper


class UserInteractionDataset(Dataset):
    def __init__(self, **kwargs):
        self.subset_name = kwargs.get("subset_name")
        self.split = kwargs.get("split", "train")
        self.neg_sample_num = kwargs.get("neg_sample_num", 4)
        max_history_size = kwargs.get("max_history_size", 50)
        self.impression = load_dataset_from_csv(f"{self.split}_{self.subset_name}")
        self.uid = np.array(self.impression["uid"], dtype=np.int32)
        if self.split == "train":
            self.positive = np.array(self.impression["positive"], dtype=np.int32)
            self.negative = [[int(n) for n in neg.split()] for neg in list(self.impression["negative"])]
        else:
            impressions = [imp.split() for imp in list(self.impression["impressions"])]
            self.candidate_nid = [np.asarray([i.split("-")[0] for i in imp], dtype=np.int32) for imp in impressions]
            # -1 for unlabeled data
            self.label = [[int(i.split("-")[1]) if "-" in i else -1 for i in imp] for imp in impressions]
        self.user_history_mapper = load_user_history_mapper(**kwargs)

    def __getitem__(self, index):
        uid = self.uid[index]
        history_nid = self.user_history_mapper[uid]
        history_mask = np.asarray(history_nid != 0, dtype=np.int8)
        if self.split == "train":
            pos = self.positive[index]
            neg = self.negative[index]
            candidate_nid = np.asarray([pos] + news_sampling(neg, self.neg_sample_num), dtype=np.int32)
            label = [1] + [0] * self.neg_sample_num
        else:
            candidate_nid = self.candidate_nid[index]
            label = self.label[index]
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
