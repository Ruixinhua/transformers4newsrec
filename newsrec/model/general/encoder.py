import torch

import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch_geometric.nn import GatedGraphConv, GAT, GATConv, GraphSAGE, Sequential

from .layer import MultiHeadAttentionAdv, AttLayer, BiAttentionLayer, RMSNorm


class MHAEncoder(nn.Module):
    def __init__(self, **model_args):
        """
        Multi-head attention encoder.
        :param model_args: head_num, head_dim, input_feat_dim (input feature dim), output_feat_dim (output feature dim)
        """
        use_flash_att = model_args.get("use_flash_att", False)
        self.input_feat_dim = model_args["input_feat_dim"]
        super(MHAEncoder, self).__init__()
        self.mha_layer = MultiHeadAttentionAdv(
            model_args["head_num"], model_args["head_dim"], self.input_feat_dim, use_flash_att=use_flash_att
        )
        output_feat_dim = model_args["head_num"] * model_args["head_dim"]  # output feature dim of MHA
        self.output_feat_dim = model_args.get("output_feat_dim", output_feat_dim)  # default output feature dim
        if model_args.get("add_linear") or self.output_feat_dim != output_feat_dim:
            self.output_layer = nn.Sequential(nn.Linear(output_feat_dim, self.output_feat_dim), nn.ReLU())

    def forward(self, x, x_mask=None) -> tuple:
        """
        Encode input vector by MHA.
        :param x: input tensor, shape = (B, F, E), B is batch size, F is the number of features, E is the embedding dim
        :param x_mask: mask tensor, shape = (B, F), 1 for valid feature, 0 for padding
        :return:
        output tensor, shape = (B, F, D), where D is head_num * head_dim by default
        output weight, shape = (B, F, F)
        """
        y, weight = self.mha_layer(x, x, x, x_mask)
        if hasattr(self, "output_layer"):
            y = self.output_layer(y)
        return y, weight


class AttLayerEncoder(nn.Module):
    def __init__(self, input_feat_dim, attention_hidden_dim):
        """
        Additive attention layer encoder.
        :param input_feat_dim: input feature dim
        :param attention_hidden_dim: hidden dim of attention layer
        """
        super(AttLayerEncoder, self).__init__()
        self.input_feat_dim = input_feat_dim
        self.output_feat_dim = input_feat_dim
        self.att_layer = AttLayer(self.input_feat_dim, attention_hidden_dim)

    def forward(self, x, x_mask=None) -> tuple:
        """
        Encode input vector by attention pooling.
        :param x: input tensor, shape = (B, F, E), B is batch size, F is the number of features, E is the embedding dim
        :param x_mask: mask tensor, shape = (B, F), 1 for valid feature, 0 for padding
        :return: output tensor, shape = (B, E)
        """
        y, weight = self.att_layer(x, x_mask)
        return y, weight


class CNNEncoder(nn.Module):
    def __init__(self, input_feat_dim, output_feat_dim, window_size):
        """
        CNN encoder.
        :param input_feat_dim: input feature dim
        :param output_feat_dim: output feature dim
        :param window_size: window size of CNN
        """
        super(CNNEncoder, self).__init__()
        self.input_feat_dim = input_feat_dim
        self.output_feat_dim = output_feat_dim  # num_filters for CNN
        padding = (window_size - 1) // 2
        assert 2 * padding == window_size - 1, "Kernel size must be an odd number"
        self.cnn_layer = nn.Conv1d(self.input_feat_dim, self.output_feat_dim, window_size)

    def forward(self, x, x_mask=None) -> tuple:
        """
        Encode input vector by CNN.
        :param x: input tensor, shape = (B, F, E), B is batch size, F is the number of features, E is the embedding dim
        :param x_mask: mask tensor, shape = (B, F), 1 for valid feature, 0 for padding
        :return: output tensor, shape = (B, F, D), where D is the number of filters
        """
        y = self.cnn_layer(x.transpose(1, 2)).transpose(1, 2)
        return y


class GRUEncoder(nn.Module):
    def __init__(self, input_feat_dim, output_feat_dim, return_all_hidden=False):
        """
        GRU encoder.
        :param input_feat_dim: input feature dim
        :param output_feat_dim: output feature dim
        :param return_all_hidden: return all hidden states or not
        """
        super(GRUEncoder, self).__init__()
        # define a variable to control the return of hidden state: last hidden or all hidden states
        self.return_all_hidden = return_all_hidden
        self.input_feat_dim = input_feat_dim
        self.output_feat_dim = output_feat_dim  # hidden_size of GRU
        self.gru_layer = nn.GRU(input_feat_dim, output_feat_dim, batch_first=True, bidirectional=False)

    def forward(self, x, x_mask, init_hidden=None) -> tuple:
        """
        Encode input vector by GRU.
        :param x: input tensor, shape = (B, F, E), B is batch size, F is the number of features, E is the embedding dim
        :param x_mask: mask tensor, shape = (B, F), 1 for valid feature, 0 for padding
        :param init_hidden: initial hidden state, shape = (1, B, D), where 1 is the number of layers
        :return:
        if return_all_hidden is True,
            output tensor, shape = (B, F, D), where D is the output feature dim
        else,
            output tensor, shape = (B, D), where D is the output feature dim, only the last hidden state
        """
        x_length = torch.sum(x_mask, dim=-1).cpu()
        x_length[x_length == 0] = 1  # avoid zero history recording
        packed_x = pack_padded_sequence(x, x_length, batch_first=True, enforce_sorted=False)
        y, y_last = self.gru_layer(packed_x, init_hidden)
        if self.return_all_hidden:
            return y
        return y_last.squeeze(0)


class BATMEncoder(nn.Module):
    def __init__(self, **model_args):
        """
        BATM encoder.
        :param model_args: topic_num, topic_dim, input_feat_dim (input feature dim), output_feat_dim
        """
        super(BATMEncoder, self).__init__()
        self.input_feat_dim = model_args.pop("input_feat_dim")
        self.output_feat_dim = model_args.pop("output_feat_dim", self.input_feat_dim)
        self.bi_attention = BiAttentionLayer(input_feat_dim=self.input_feat_dim, output_feat_dim=self.output_feat_dim,
                                             **model_args)

    def forward(self, x, x_mask) -> tuple:
        """
        Encode input vector by BATM.
        :param x: input tensor, shape = (B, F, E), B is batch size, F is the number of features, E is the embedding dim
        :param x_mask: mask tensor, shape = (B, F), 1 for valid feature, 0 for padding
        :return:
        output tensor, shape = (B, F, E)
        output weight, shape = (B, F)
        """
        topic_dict = self.bi_attention(x, x_mask)
        return topic_dict["topic_vector"], topic_dict["topic_weight"]


class UserIDEncoder(nn.Module):
    def __init__(self, user_num, uid_embed_dim=100):
        """
        User ID encoder, encode user ID to user embedding.
        :param user_num: number of users
        :param uid_embed_dim: user embedding dim, by default 100
        """
        super(UserIDEncoder, self).__init__()
        self.uid_embedd_dim = uid_embed_dim
        self.uid_embedding = nn.Embedding(user_num + 1, self.uid_embedd_dim)
        self.output_feat_dim = self.uid_embedd_dim

    def forward(self, x) -> torch.Tensor:
        """
        Encode user ID to user embedding.
        :param x: input tensor, shape = (B), B is batch size
        :return: output_tensor, shape = (B, D), where D is the output feature dim
        """
        return self.uid_embedding(x)


class UserEncoderGRU(nn.Module):
    def __init__(self, user_num, user_encoder_name, input_feat_dim, output_feat_dim, attention_hidden_dim):
        """
        User encoder by GRU.
        :param user_num: number of users
        :param user_encoder_name: user encoder name, e.g., last_gru, init_gru, concat_gru, without_uid_gru
        :param input_feat_dim: input feature dim
        :param output_feat_dim: output feature dim
        :param attention_hidden_dim: hidden dim of attention
        """
        super(UserEncoderGRU, self).__init__()
        self.input_feat_dim, self.output_feat_dim = input_feat_dim, output_feat_dim
        self.user_encoder_name = user_encoder_name
        self.user_embedding = UserIDEncoder(user_num)
        if "last" in user_encoder_name:
            self.return_all_hidden = False
        else:
            self.return_all_hidden = True
        if "init" in user_encoder_name:  # init_gru_last/concat_gru_last/gru_without_uid_last/concat_gru/
            self.user_embed_method = "init"
        elif "concat" in user_encoder_name:
            self.user_embed_method = "concat"
        elif "without_uid" in user_encoder_name:
            self.user_embed_method = "without_uid"
        self.gru_layer = GRUEncoder(input_feat_dim=self.input_feat_dim, output_feat_dim=self.output_feat_dim,
                                    return_all_hidden=self.return_all_hidden)
        if self.return_all_hidden:
            self.aggregate_attention = AttLayerEncoder(input_feat_dim=self.gru_layer.output_feat_dim,
                                                       attention_hidden_dim=attention_hidden_dim)
        uid_embedd_dim = self.user_embedding.uid_embedd_dim
        if self.user_embed_method == "concat":
            self.transform_layer = nn.Sequential(
                nn.Linear(self.gru_layer.output_feat_dim + uid_embedd_dim, self.output_feat_dim), nn.ReLU()
            )
        elif self.user_embed_method == "init":
            self.user_affine = nn.Sequential(nn.Linear(uid_embedd_dim, self.input_feat_dim), nn.ReLU())

    def forward(self, x, x_mask, uid) -> tuple:
        """
        Encode user by GRU.
        :param x: shape, shape = (B, H, E), B is batch size, H is the number of history news, E is the embedding dim
        :param x_mask: mask tensor, shape = (B, H), 1 for valid feature, 0 for padding
        :param uid: user ID, shape = (B)
        :return: output tensor, shape = (B, D), where D is the output feature dim
        """
        user_embed, user_weight = self.user_embedding(uid), None
        if self.user_embed_method == "concat":
            y = self.gru_layer(x, x_mask)
            if self.return_all_hidden:
                y = pad_packed_sequence(y, batch_first=True)[0]
                if "LF" in self.user_encoder_name:  # late fusion: return all hidden states
                    return y, user_weight
                y, user_weight = self.aggregate_attention(y)
            y = self.transform_layer(torch.cat([y, user_embed], dim=1))  # concatenate user embedding
        else:
            if self.user_embed_method == "init":
                user_embed = self.user_affine(user_embed)
                y = self.gru_layer(x, x_mask, init_hidden=user_embed.unsqueeze(0))
            else:
                y = self.gru_layer(x, x_mask)
            if self.return_all_hidden:
                y = pad_packed_sequence(y, batch_first=True)[0]
                if "LF" in self.user_encoder_name:  # late fusion: return all hidden states
                    return y, user_weight
                y, user_weight = self.aggregate_attention(y)
        return y, user_weight


class GraphEncoder(nn.Module):
    def __init__(self, graph, neighbors_num, **model_args):
        """
        Graph encoder for news graph and entity graph.
        :param graph: graph object
        :param neighbors_num: number of neighbors
        :param model_args: gnn_model, input_feat_dim, output_feat_dim
        """
        super(GraphEncoder, self).__init__()
        neighbor_matrix = torch.zeros((len(graph.neighbor_mapper)+1, neighbors_num), dtype=torch.int32)
        for k, v in graph.neighbor_mapper.items():
            neighbor_matrix[k, :min(neighbors_num, len(v))] = torch.tensor(v[:neighbors_num])
        self.neighbors = nn.Embedding.from_pretrained(neighbor_matrix, freeze=True)
        self.gnn_model = model_args["gnn_model"]
        self.input_feat_dim, self.output_feat_dim = model_args["input_feat_dim"], model_args["output_feat_dim"]
        if self.gnn_model.lower() == "gat":
            self.gnn_encoder = GAT(self.input_feat_dim, out_channels=self.output_feat_dim,
                                   hidden_channels=64, num_layers=3)
        elif self.gnn_model.lower() == "gatconv":
            self.gnn_encoder = GATConv(self.input_feat_dim, out_channels=self.output_feat_dim)
        elif self.gnn_model.lower() == "graphsage":
            self.gnn_encoder = GraphSAGE(self.input_feat_dim, 64, 3, self.output_feat_dim)
        else:
            self.gnn_encoder = GatedGraphConv(self.output_feat_dim, num_layers=3, aggr="add")

    def forward(self, x, index):
        return self.gnn_encoder(x, index)


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ResidualBlock, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = nn.LeakyReLU(0.2)

        if input_dim != output_dim:
            self.shortcut = nn.Linear(input_dim, output_dim)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.linear(x)
        out = self.activation(out)
        out += identity
        return out


class GLUAggregator(nn.Module):
    def __init__(self, embedding_dim):
        super(GLUAggregator, self).__init__()
        self.linear = nn.Linear(embedding_dim, embedding_dim * 2)

    def forward(self, features_list):
        aggregated_list = []
        for feature in features_list:
            gated = self.linear(feature)  # Shape: (batch_size, num_history, embedding_dim * 2)
            value, gate = gated.chunk(2, dim=-1)  # Split into value and gate
            aggregated = value * torch.sigmoid(gate)  # Apply gating
            aggregated_list.append(aggregated)

        # Sum all gated outputs
        output = torch.stack(aggregated_list, dim=0).sum(dim=0)
        return output  # Shape: (batch_size, num_history, embedding_dim)


class NaiveFeatureInteractionAggregator(nn.Module):
    def __init__(self, input_feat_dim):
        super(NaiveFeatureInteractionAggregator, self).__init__()
        # MLP for feature interaction
        self.mlp = nn.Sequential(
            nn.Linear(input_feat_dim, input_feat_dim),
            nn.ReLU(),
            nn.Linear(input_feat_dim, input_feat_dim)
        )

    def forward(self, features_list):
        # Concatenate all features along the embedding dimension
        features_cat = torch.cat(features_list, dim=-1)
        # Shape: (batch_size, num_history, num_features * embedding_dim)

        # Apply MLP for feature interaction
        interacted_features = self.mlp(features_cat)
        # Shape: (batch_size, num_history, embedding_dim)

        # Residual connection to preserve original information
        output = features_cat + interacted_features
        # Shape: (batch_size, num_history, embedding_dim)

        return output


class FeatureInteractionAggregator(nn.Module):
    def __init__(self, input_feat_dim, output_feat_dim, num_features):
        super(FeatureInteractionAggregator, self).__init__()

        # Attention mechanism to compute feature importance
        self.attention = nn.Linear(input_feat_dim, num_features)

        # MLP for feature interaction
        self.mlp = nn.Sequential(
            nn.Linear(input_feat_dim, output_feat_dim),
            nn.ReLU(),
            nn.Linear(output_feat_dim, output_feat_dim)
        )

    def forward(self, features_list):
        # Concatenate all features along the embedding dimension
        features_cat = torch.cat(features_list, dim=-1)
        # Shape: (batch_size, num_history, num_features * embedding_dim)

        # Attention mechanism to get weights for each feature
        weights = torch.softmax(self.attention(features_cat), dim=-1)
        # Shape: (batch_size, num_history, num_features)
        weights = weights.unsqueeze(-1)  # Shape: (batch_size, num_history, num_features, 1)

        # Stack features and apply attention weights
        features_stacked = torch.stack(features_list, dim=2)
        # Shape: (batch_size, num_history, num_features, embedding_dim)

        weighted_features = weights * features_stacked
        # Shape: (batch_size, num_history, num_features, embedding_dim)

        aggregated_features = torch.sum(weighted_features, dim=2)
        # Shape: (batch_size, num_history, embedding_dim)

        # Apply MLP for feature interaction
        interacted_features = self.mlp(features_cat)
        # Shape: (batch_size, num_history, embedding_dim)

        # Residual connection to preserve original information
        output = aggregated_features + interacted_features
        # Shape: (batch_size, num_history, embedding_dim)

        return output


class NormFeatureInteractionAggregator(nn.Module):
    def __init__(self, input_feat_dim, output_feat_dim, num_features):
        super(NormFeatureInteractionAggregator, self).__init__()

        # Attention mechanism to compute feature importance
        self.attention = nn.Linear(input_feat_dim, num_features)
        self.attention_norm = RMSNorm(input_feat_dim)

        # MLP for feature interaction
        self.mlp = nn.Sequential(
            nn.Linear(input_feat_dim, output_feat_dim),
            nn.ReLU(),
            nn.Linear(output_feat_dim, output_feat_dim)
        )
        self.mlp_norm = RMSNorm(input_feat_dim)

    def forward(self, features_list):
        # Concatenate all features along the embedding dimension
        features_cat = torch.cat(features_list, dim=-1)
        # Shape: (batch_size, num_history, num_features * embedding_dim)

        # Attention mechanism to get weights for each feature
        weights = torch.softmax(self.attention(self.attention_norm(features_cat)), dim=-1)
        # Shape: (batch_size, num_history, num_features)
        weights = weights.unsqueeze(-1)  # Shape: (batch_size, num_history, num_features, 1)

        # Stack features and apply attention weights
        features_stacked = torch.stack(features_list, dim=2)
        # Shape: (batch_size, num_history, num_features, embedding_dim)

        weighted_features = weights * features_stacked
        # Shape: (batch_size, num_history, num_features, embedding_dim)

        aggregated_features = torch.sum(weighted_features, dim=2)
        # Shape: (batch_size, num_history, embedding_dim)

        # Apply MLP for feature interaction
        interacted_features = self.mlp(self.mlp_norm(features_cat))
        # Shape: (batch_size, num_history, embedding_dim)

        # Residual connection to preserve original information
        output = aggregated_features + interacted_features
        # Shape: (batch_size, num_history, embedding_dim)

        return output


class NaiveNormSequentialFeatureAggregator(nn.Module):
    def __init__(self, input_feat_dim, num_features):
        super(NaiveNormSequentialFeatureAggregator, self).__init__()
        self.input_feat_dim = input_feat_dim
        self.output_feat_dim = input_feat_dim * num_features

        # Aggregator for two features at a time
        self.aggregator = nn.ModuleList([
            nn.Sequential(
                NaiveFeatureInteractionAggregator(input_feat_dim),  # NaiveFeatureInteractionAggregator
                RMSNorm(input_feat_dim)  # RMSNorm layer
            ) for _ in range(num_features)
        ])

    def forward(self, features_list):
        # Step-wise aggregation of features
        aggregated = torch.cat([agg([features_list[i]]) for i, agg in enumerate(self.aggregator)], dim=-1)
        return aggregated


class EnhancedFeatureInteractionAggregator(nn.Module):
    def __init__(self, input_feat_dim, output_feat_dim, num_features):
        super(EnhancedFeatureInteractionAggregator, self).__init__()

        # Attention mechanism to compute feature importance
        self.attention = nn.Linear(input_feat_dim, num_features)

        # Gating mechanism to control feature contribution
        self.gate = nn.Linear(input_feat_dim, num_features)

        # MLP for feature interaction
        self.mlp = nn.Sequential(
            nn.Linear(input_feat_dim, output_feat_dim),
            nn.ReLU(),
            nn.Linear(output_feat_dim, output_feat_dim)
        )

    def forward(self, features_list):
        # Concatenate all features along the embedding dimension
        features_cat = torch.cat(features_list, dim=-1)
        # Shape: (batch_size, num_history, num_features * embedding_dim)

        # Attention mechanism to get weights for each feature
        weights = torch.softmax(self.attention(features_cat), dim=-1)
        # Shape: (batch_size, num_history, num_features)
        weights = weights.unsqueeze(-1)  # Shape: (batch_size, num_history, num_features, 1)

        # Gating mechanism to determine the importance of each feature
        gates = torch.sigmoid(self.gate(features_cat))  # Shape: (batch_size, num_history, num_features)
        gates = gates.unsqueeze(-1)  # Shape: (batch_size, num_history, num_features, 1)

        # Stack features and apply attention weights and gates
        features_stacked = torch.stack(features_list, dim=2)
        # Shape: (batch_size, num_history, num_features, embedding_dim)

        weighted_features = weights * gates * features_stacked
        # Shape: (batch_size, num_history, num_features, embedding_dim)

        aggregated_features = torch.sum(weighted_features, dim=2)
        # Shape: (batch_size, num_history, embedding_dim)

        # Apply MLP for feature interaction
        interacted_features = self.mlp(features_cat)
        # Shape: (batch_size, num_history, embedding_dim)

        # Residual connection to preserve original information
        output = aggregated_features + interacted_features
        # Shape: (batch_size, num_history, embedding_dim)

        return output


class SequentialFeatureAggregator(nn.Module):
    def __init__(self, output_feat_dim):
        super(SequentialFeatureAggregator, self).__init__()
        self.output_feat_dim = output_feat_dim

        # Aggregator for two features at a time
        self.aggregator = FeatureInteractionAggregator(output_feat_dim * 2, output_feat_dim, 2)

    def forward(self, features_list):
        # Step-wise aggregation of features
        aggregated = features_list[0]
        for i in range(1, len(features_list)):
            aggregated = self.aggregator([aggregated, features_list[i]])

        return aggregated


class EnhancedSequentialFeatureAggregator(nn.Module):
    def __init__(self, output_feat_dim):
        super(EnhancedSequentialFeatureAggregator, self).__init__()
        self.output_feat_dim = output_feat_dim

        # Enhanced Aggregator for two features at a time
        self.aggregator = EnhancedFeatureInteractionAggregator(output_feat_dim * 2, output_feat_dim, 2)

    def forward(self, features_list):
        # Step-wise aggregation of features
        aggregated = features_list[0]
        for i in range(1, len(features_list)):
            aggregated = self.aggregator([aggregated, features_list[i]])

        return aggregated


class MultiHeadFeatureInteractionAggregator(nn.Module):
    def __init__(self, input_feat_dim, output_feat_dim, num_features):
        super(MultiHeadFeatureInteractionAggregator, self).__init__()

        # Multi-head attention mechanism to compute feature importance
        self.multi_head_attention = nn.MultiheadAttention(embed_dim=input_feat_dim, num_heads=4, batch_first=True)

        # Gating mechanism to control feature contribution
        self.gate = nn.Linear(input_feat_dim, num_features)

        # MLP for feature interaction
        self.mlp = nn.Sequential(
            nn.Linear(input_feat_dim, output_feat_dim),
            nn.ReLU(),
            nn.Linear(output_feat_dim, output_feat_dim)
        )

    def forward(self, features_list):
        # Concatenate all features along the embedding dimension
        features_cat = torch.cat(features_list, dim=-1)
        # Shape: (batch_size, num_history, num_features * embedding_dim)

        # Multi-head attention to get weighted features
        attn_output, _ = self.multi_head_attention(features_cat, features_cat, features_cat)
        # Shape: (batch_size, num_history, num_features * embedding_dim)

        # Gating mechanism to determine the importance of each feature
        gates = torch.sigmoid(self.gate(features_cat))  # Shape: (batch_size, num_history, num_features)
        gates = gates.unsqueeze(-1)  # Shape: (batch_size, num_history, num_features, 1)

        # Stack features and apply attention weights and gates
        features_stacked = torch.stack(features_list, dim=2)
        # Shape: (batch_size, num_history, num_features, embedding_dim)

        weighted_features = gates * features_stacked
        # Shape: (batch_size, num_history, num_features, embedding_dim)

        aggregated_features = torch.sum(weighted_features, dim=2)
        # Shape: (batch_size, num_history, embedding_dim)

        # Apply MLP for feature interaction
        interacted_features = self.mlp(attn_output)
        # Shape: (batch_size, num_history, embedding_dim)

        # Residual connection to preserve original information
        output = aggregated_features + interacted_features
        # Shape: (batch_size, num_history, embedding_dim)

        return output


class FusionAggregator(nn.Module):
    def __init__(self, input_feat_dim, output_feat_dim, num_features, agg_method='mlp'):
        """
        Fusion aggregator for feature vectors.
        :param input_feat_dim: embedding dimension of each feature vectors
        :param output_feat_dim: output feature dim
        :param num_features: number of feature vectors
        :param agg_method: aggregation method, mlp or att
        Input shape: List of Tensors(batch_size, feature_num, news_num, input_feat_dim)
        """
        super().__init__()
        self.input_feat_dim = input_feat_dim
        self.output_feat_dim = output_feat_dim
        self.agg_method = agg_method
        ft_methods = {
            "feature_interaction": FeatureInteractionAggregator,
            "norm_feature_interaction": NormFeatureInteractionAggregator,
            "enhanced_feature_interaction": EnhancedFeatureInteractionAggregator,
        }
        if agg_method == "mlp":
            self.aggregator = Sequential('a', [
                (lambda a: torch.cat(a, dim=-1), f'a -> x'),
                nn.Linear(self.input_feat_dim, int(self.output_feat_dim * 2)),
                nn.LeakyReLU(0.2),
                nn.Linear(int(self.output_feat_dim * 2), int(self.output_feat_dim * 1.5)),
                nn.LeakyReLU(0.2),
                nn.Linear(int(self.output_feat_dim * 1.5), self.output_feat_dim),
            ])
        elif agg_method == "mlp_new":
            self.aggregator = Sequential('a', [
                (lambda a: torch.cat(a, dim=-1), f'a -> x'),
                nn.Linear(self.input_feat_dim, int(self.input_feat_dim / 2)),
                nn.LeakyReLU(0.2),
                nn.Linear(int(self.input_feat_dim / 2), int(self.input_feat_dim / 4)),
                nn.LeakyReLU(0.2),
                nn.Linear(int(self.input_feat_dim / 4), self.output_feat_dim),
            ])
        elif agg_method == "mlp_gelu":
            self.aggregator = Sequential('a', [
                (lambda a: torch.cat(a, dim=-1), f'a -> x'),
                nn.Linear(self.input_feat_dim, int(self.input_feat_dim / 2)),
                nn.GELU(),
                nn.Linear(int(self.input_feat_dim / 2), int(self.input_feat_dim / 4)),
                nn.GELU(),
                nn.Linear(int(self.input_feat_dim / 4), self.output_feat_dim),
            ])
        elif agg_method == "mlp_relu":
            self.aggregator = Sequential('a', [
                (lambda a: torch.cat(a, dim=-1), f'a -> x'),
                nn.Linear(self.input_feat_dim, int(self.input_feat_dim / 2)),
                nn.ReLU(),
                nn.Linear(int(self.input_feat_dim / 2), int(self.input_feat_dim / 4)),
                nn.ReLU(),
                nn.Linear(int(self.input_feat_dim / 4), self.output_feat_dim),
            ])
        elif agg_method == "mlp_residual":
            self.aggregator = Sequential('a', [
                (lambda a: torch.cat(a, dim=-1), 'a -> x'),
                ResidualBlock(self.input_feat_dim, int(self.input_feat_dim / 2)),
                ResidualBlock(int(self.input_feat_dim / 2), int(self.input_feat_dim / 4)),
                ResidualBlock(int(self.input_feat_dim / 4), self.output_feat_dim),
            ])
        elif agg_method == "naive_feature_interaction":
            self.aggregator = Sequential('a', [
                (NaiveFeatureInteractionAggregator(input_feat_dim), 'a -> x')
            ])
            self.output_feat_dim = input_feat_dim
        elif agg_method in ft_methods:
            self.aggregator = Sequential('a', [
                (ft_methods[agg_method](input_feat_dim, output_feat_dim, num_features), 'a -> x')
            ])
        elif agg_method == "naive_norm_seq_fi":
            self.aggregator = Sequential('a', [
                (NaiveNormSequentialFeatureAggregator(output_feat_dim, num_features), 'a -> x')
            ])
            self.output_feat_dim = output_feat_dim * num_features
        elif agg_method == "enhanced_feature_interaction_seq":
            self.aggregator = Sequential('a', [
                (EnhancedSequentialFeatureAggregator(output_feat_dim), 'a -> x')
            ])
        elif agg_method == "feature_interaction_seq":  # use multiple feature interaction layers
            self.aggregator = Sequential('a', [
                (SequentialFeatureAggregator(output_feat_dim), 'a -> x')
            ])
        elif agg_method == "glu":
            self.aggregator = Sequential('a', [
                (GLUAggregator(output_feat_dim), 'a -> x')
            ])
        elif agg_method == "cat":
            self.aggregator = Sequential('a', [
                (lambda a: torch.cat(a, dim=-1), 'a -> x')
            ])
        elif agg_method == "cat_lrelu":
            self.aggregator = Sequential('a', [
                (lambda a: torch.cat(a, dim=-1), 'a -> x'),
                nn.LeakyReLU(0.2)
            ])
        elif agg_method == "cat_equal":
            self.aggregator = Sequential('a', [
                (lambda a: torch.cat(a, dim=-1), 'a -> x'),
                nn.Linear(self.input_feat_dim, self.input_feat_dim),
                nn.LeakyReLU(0.2)
            ])
            self.output_feat_dim = input_feat_dim
        elif agg_method == "cat_mlp":
            self.aggregator = Sequential('a', [
                (lambda a: torch.cat(a, dim=-1), 'a -> x'),
                nn.Linear(self.input_feat_dim, self.output_feat_dim),
                nn.LeakyReLU(0.2)
            ])
        else:
            raise ValueError(f"Invalid aggregation method: {agg_method}")

    def forward(self, vectors):
        return self.aggregator(vectors)
