from newsrec.model.general import BATMEncoder, AttLayerEncoder, MHAEncoder, UserEncoderGRU, CNNEncoder
from newsrec.model.general import FusionAggregator, LateFusion
from .base import BaseNRS


class BATMNRModel(BaseNRS):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        default_args = {
            "head_num": 20, "head_dim": 20, "topic_num": 50, "topic_dim": 20, "news_encoder_name": "batm_att",
            "user_encoder_name": "batm_att", "attention_hidden_dim": 200, "topic_layer_name": "base_att",
            "window_size": 3, "num_filters": 300, "fusion_method": "weighted", "add_news_dropout": False
        }
        for key, value in default_args.items():
            kwargs.setdefault(key, value)
        self.news_encoder_name, self.user_encoder_name = kwargs["news_encoder_name"], kwargs["user_encoder_name"]
        self.add_news_dropout = kwargs["add_news_dropout"]
        if self.news_encoder_name == "batm_att":
            self.news_encode_layer = BATMEncoder(input_feat_dim=self.embedding_dim, **kwargs)
            # in: (B, F, E), out: (B, F, E)
            self.news_layer = AttLayerEncoder(input_feat_dim=self.news_encode_layer.output_feat_dim,
                                              attention_hidden_dim=kwargs["attention_hidden_dim"])
            # in: (B, F, E), out: (B, E)
        elif self.news_encoder_name == "batm_mha_att":
            self.news_encode_layer = BATMEncoder(input_feat_dim=self.embedding_dim, **kwargs)
            # in: (B, F, E), out: (B, F, E)
            self.news_encode_layer2 = MHAEncoder(input_feat_dim=self.news_encode_layer.output_feat_dim, **kwargs)
            # in: (B, F, E), out: (B, F, D); D = head_num * head_dim
            self.news_layer = AttLayerEncoder(input_feat_dim=self.news_encode_layer2.output_feat_dim,
                                              attention_hidden_dim=kwargs["attention_hidden_dim"])
            # in: (B, F, D), out: (B, D)
        elif self.news_encoder_name == "mha_att":
            self.news_encode_layer = MHAEncoder(input_feat_dim=self.embedding_dim, **kwargs)
            # in: (B, F, E), out: (B, F, D); D = head_num * head_dim
            self.news_layer = AttLayerEncoder(input_feat_dim=self.news_encode_layer.output_feat_dim,
                                              attention_hidden_dim=kwargs["attention_hidden_dim"])
            # in: (B, F, D), out: (B, E)
        elif self.news_encoder_name == "cnn_att":
            self.news_encode_layer = CNNEncoder(
                input_feat_dim=self.embedding_dim, output_feat_dim=kwargs["num_filters"],
                window_size=kwargs["window_size"]
            )
            # in: (B, F, E), out: (B, F, D)
            self.news_layer = AttLayerEncoder(input_feat_dim=self.news_encode_layer.output_feat_dim,
                                              attention_hidden_dim=kwargs["attention_hidden_dim"])
            # in: (B, F, D), out: (B, E)
        elif self.news_encoder_name == "batm_cnn_att":
            self.news_encode_layer = BATMEncoder(input_feat_dim=self.embedding_dim, **kwargs)
            # in: (B, F, E), out: (B, F, E)
            self.news_encode_layer2 = CNNEncoder(
                input_feat_dim=self.news_encode_layer.output_feat_dim, output_feat_dim=kwargs["num_filters"],
                window_size=kwargs["window_size"]
            )
            # in: (B, F, E), out: (B, F, D)
            self.news_layer = AttLayerEncoder(input_feat_dim=self.news_encode_layer2.output_feat_dim,
                                              attention_hidden_dim=kwargs["attention_hidden_dim"])
            # in: (B, F, D), out: (B, E)
        else:
            raise ValueError(f"news_encoder_name {self.news_encoder_name} is not supported")
        # self.news_layer is the final layer for news encoder

        if self.user_encoder_name == "batm_att":
            self.user_encode_layer = BATMEncoder(
                input_feat_dim=self.news_layer.output_feat_dim, output_feat_dim=self.news_layer.output_feat_dim,
                **kwargs)
            # in: (B, F, E), out: (B, F, E)
            self.user_layer = AttLayerEncoder(input_feat_dim=self.news_layer.output_feat_dim,
                                              attention_hidden_dim=kwargs["attention_hidden_dim"])
            # in: (B, F, E), out: (B, E)
        elif self.user_encoder_name == "batm_att_LF":
            self.user_encode_layer = BATMEncoder(
                input_feat_dim=self.news_layer.output_feat_dim, output_feat_dim=self.news_layer.output_feat_dim,
                **kwargs)
            self.late_fusion = LateFusion(kwargs["fusion_method"], input_feat_dim=self.news_layer.output_feat_dim)
            # in: (B, F, E), out: (B, F, E)
        elif "feature_interaction" in self.user_encoder_name:
            self.user_encode_layer = FusionAggregator(
                self.news_layer.output_feat_dim, self.news_layer.output_feat_dim, 1, self.user_encoder_name
            )
            self.cand_encode_layer = FusionAggregator(
                self.news_layer.output_feat_dim, self.news_layer.output_feat_dim, 1, self.user_encoder_name
            )
            self.late_fusion = LateFusion(kwargs["fusion_method"], input_feat_dim=self.news_layer.output_feat_dim)
        elif self.user_encoder_name == "att":
            self.user_layer = AttLayerEncoder(input_feat_dim=self.news_layer.output_feat_dim,
                                              attention_hidden_dim=kwargs["attention_hidden_dim"])
            # in: (B, F, E), out: (B, E)
        elif self.user_encoder_name == "mha_att":
            self.user_encode_layer = MHAEncoder(
                input_feat_dim=self.news_layer.output_feat_dim, output_feat_dim=self.news_layer.output_feat_dim,
                **kwargs)
            # in: (B, F, E), out: (B, F, D); D = head_num * head_dim
            self.user_layer = AttLayerEncoder(input_feat_dim=self.news_layer.output_feat_dim,
                                              attention_hidden_dim=kwargs["attention_hidden_dim"])
            # in: (B, F, D), out: (B, E)
        elif self.user_encoder_name == "mha_att_LF":
            self.user_encode_layer = MHAEncoder(
                input_feat_dim=self.news_layer.output_feat_dim, output_feat_dim=self.news_layer.output_feat_dim,
                **kwargs)
            self.late_fusion = LateFusion(kwargs["fusion_method"], input_feat_dim=self.news_layer.output_feat_dim)
            # in: (B, F, E), out: (B, F, D); D = head_num * head_dim
        elif self.user_encoder_name == "batm_mha_att":
            self.user_encode_layer = BATMEncoder(input_feat_dim=self.news_layer.output_feat_dim, **kwargs)
            # in: (B, F, E), out: (B, F, E)
            self.user_encode_layer2 = MHAEncoder(
                # in: (B, F, E), out: (B, F, D); D = news_layer.output_feat_dim
                input_feat_dim=self.user_encode_layer.output_feat_dim, output_feat_dim=self.news_layer.output_feat_dim,
                **kwargs
            )
            # in: (B, F, E), out: (B, F, D); D = head_num * head_dim
            self.user_layer = AttLayerEncoder(input_feat_dim=self.news_layer.output_feat_dim,
                                              attention_hidden_dim=kwargs["attention_hidden_dim"])
            # in: (B, F, D), out: (B, D)
        elif "gru" in self.user_encoder_name:
            if "LF" in self.user_encoder_name:
                self.late_fusion = LateFusion(kwargs["fusion_method"], input_feat_dim=self.news_layer.output_feat_dim)
            self.user_encoder_layer = UserEncoderGRU(
                input_feat_dim=self.news_layer.output_feat_dim, output_feat_dim=self.news_layer.output_feat_dim,
                user_num=len(self.user_history), user_encoder_name=self.user_encoder_name,
                attention_hidden_dim=kwargs["attention_hidden_dim"]
            )
            # in: (B, F, E), out: (B, E)
        else:
            raise ValueError(f"user_encoder_name {self.user_encoder_name} is not supported")

    def feature_extractor(self, input_feat):
        word_vector, news_mask = self.text_feature_encoder(input_feat)  # shape = (B*(H+C), F, E)
        news_mask = None
        if self.use_news_mask:
            news_mask = news_mask
        if self.add_news_dropout:
            word_vector = self.dropout_ne(word_vector)
        return self.news_encode_layer(word_vector, news_mask)

    def news_encoder(self, input_feat):
        """input_feat: Size is [N * H, S]"""
        if self.news_encoder_name == "batm_att":
            # extract topic vector--in: (B, F, E), out: (B, F, E)
            topic_vector, topic_weight = self.feature_extractor(input_feat)
            # add additive attention layer--in: (B, F, E), out: (B, E)
            news_vector, news_weight = self.news_layer(topic_vector)
        elif self.news_encoder_name == "batm_mha_att":
            # extract topic vector--in: (B, F, E), out: (B, F, E)
            topic_vector, topic_weight = self.feature_extractor(input_feat)
            # add multi-head attention layer--in: (B, F, E), out: (B, F, E)
            topic_vector, _ = self.news_encode_layer2(topic_vector)
            # add additive attention layer--in: (B, F, E), out: (B, E)
            news_vector, news_weight = self.news_layer(topic_vector)
        elif self.news_encoder_name == "mha_att":
            # add multi-head attention layer--in: (B, F, E), out: (B, F, D)
            y, topic_weight = self.feature_extractor(input_feat)
            # add additive attention layer--in: (B, F, D), out: (B, E)
            news_vector, news_weight = self.news_layer(y)
        elif self.news_encoder_name == "cnn_att":
            # add CNN layer--in: (B, F, E), out: (B, F, D)
            y = self.feature_extractor(input_feat)
            # add additive attention layer--in: (B, F, D), out: (B, E)
            news_vector, news_weight = self.news_layer(y)
            topic_weight = None
        elif self.news_encoder_name == "batm_cnn_att":
            # extract topic vector--in: (B, F, E), out: (B, F, E)
            topic_vector, topic_weight = self.feature_extractor(input_feat)
            # add CNN layer--in: (B, F, E), out: (B, F, D)
            y = self.news_encode_layer2(topic_vector)
            # add additive attention layer--in: (B, F, D), out: (B, E)
            news_vector, news_weight = self.news_layer(y)
        else:
            raise ValueError(f"news_encoder_name {self.news_encoder_name} is not supported")
        return {"news_vector": news_vector, "news_weight": news_weight, "topic_weight": topic_weight}

    def user_encoder(self, input_feat):
        user_mask, user_weight = None, None
        output = {}
        if self.use_user_mask:
            user_mask = input_feat["history_mask"]
        if "gru" in self.user_encoder_name:
            user_vector, user_weight = self.user_encoder_layer(
                input_feat["history_news"], input_feat["history_mask"], input_feat["uid"]
            )
        elif self.user_encoder_name == "batm_att":
            y, _ = self.user_encode_layer(input_feat["history_news"], user_mask)
            user_vector, user_weight = self.user_layer(y)
        elif "LF" in self.user_encoder_name:
            user_vector, user_weight = self.user_encode_layer(input_feat["history_news"], user_mask)
        elif "feature_interaction" in self.user_encoder_name:
            user_vector = self.user_encode_layer([input_feat["history_news"]])
            cand_vector = self.cand_encode_layer([input_feat["candidate_news"]])
            output["candidate_news_vector"] = cand_vector
        elif self.user_encoder_name == "att":
            user_vector, user_weight = self.user_layer(input_feat["history_news"])
        elif self.user_encoder_name == "mha_att":
            y, _ = self.user_encode_layer(input_feat["history_news"], user_mask)
            user_vector, user_weight = self.user_layer(y)
        elif self.user_encoder_name == "batm_mha_att":
            y, _ = self.user_encode_layer(input_feat["history_news"], user_mask)
            y, _ = self.user_encode_layer2(y, user_mask)
            user_vector, user_weight = self.user_layer(y)
        else:
            raise ValueError(f"user_encoder_name {self.user_encoder_name} is not supported")
        output.update({"user_vector": user_vector, "user_weight": user_weight})
        return output

    def predict(self, candidate_news_vector, user_vector, **kwargs):
        """
        prediction logic: use MLP or Dot-product for prediction
        :param candidate_news_vector: shape = (B, C, D)
        :param user_vector: shape = (B, D)
        :return: softmax possibility of click candidate news
        """
        if hasattr(self, "late_fusion"):
            score = self.late_fusion(user_vector, candidate_news_vector)
        else:
            score = self.click_predictor(candidate_news_vector, user_vector)
        return score
