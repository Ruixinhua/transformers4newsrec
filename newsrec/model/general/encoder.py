import torch.nn as nn

from .layer import MultiHeadedAttention, AttLayer, BiAttentionLayer


class NRMSEncoder(nn.Module):
    def __init__(self, head_num, head_dim, embedding_dim, attention_hidden_dim, dropout_rate=0, use_flash_att=False):
        super(NRMSEncoder, self).__init__()
        self.mha_layer = MultiHeadedAttention(head_num, head_dim, embedding_dim, use_flash_att=use_flash_att)
        # output tensor shape = (B, F, D)
        self.att_layer = AttLayer(head_num * head_dim, attention_hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, x_mask=None):
        """
        Encode input vector by MHA and attention pooling.
        :param x: input tensor, shape = (B, F, E), B is batch size, F is the number of features, E is the embedding dim
        :param x_mask: mask tensor, shape = (B, F), 1 for valid feature, 0 for padding
        :return: output tensor, shape = (B, E)
        """
        y = self.mha_layer(x, x, x, x_mask)[0]
        out = self.att_layer(self.dropout(y), x_mask)
        return {"y": out[0], "weight": out[1]}


class BiAttentionEncoder(nn.Module):
    def __init__(self, **model_args):
        super(BiAttentionEncoder, self).__init__()
        self.bi_attention = BiAttentionLayer(**model_args)
        self.att_layer = AttLayer(model_args["embedding_dim"], model_args["attention_hidden_dim"])
        self.dropout = nn.Dropout(model_args["dropout_rate"])

    def forward(self, x, x_mask):
        topic_out = self.bi_attention(x, x_mask)
        out = self.att_layer(self.dropout(topic_out["topic_vector"]))
        return {"y": out[0], "weight": out[1], "topic_weight": topic_out["topic_weight"]}
