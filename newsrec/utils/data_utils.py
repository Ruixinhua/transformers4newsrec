# -*- coding: utf-8 -*-
# @Author        : Rui
# @Time          : 2024/4/14 15:25
# @Function      : Define the utils functions processing news data
import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union
from datasets import Dataset as HFDataset
from newsrec.utils import get_project_root


def load_dataset_from_csv(data_name, cache_dir=None):
    """
    Load HFDataset object from remote csv file
    :param data_name: in the format of {data}_{subset_name}, e.g. train_small, news_small
    :param cache_dir: directory of cached csv data
    :return: HFDataset object
    """
    if cache_dir is None:
        cache_dir = f"{get_project_root()}/cached/MIND"
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_file = f"{data_name}.csv"
    if (cache_dir / data_file).exists():
        df = pd.read_csv(cache_dir / data_file)
    else:
        df = pd.read_csv(f"hf://datasets/Rui98/mind/{data_file}")
        df.to_csv(cache_dir / data_file, index=False)
    df.fillna("", inplace=True)
    dataset = HFDataset.from_pandas(df)
    return dataset


def load_embedding_from_glove_name(glove_name=None):
    """
    Load glove embedding from remote or cached parquet file
    :param glove_name: default glove name, glove_6b_300d
    :return: glove embeddings of DataFrame object
    """
    if glove_name is None:
        glove_name = "glove_6b_300d"
    glove_path = Path(f"{get_project_root()}/cached/{glove_name}.parquet")
    if glove_path.exists():
        glove_embeds = pd.read_parquet(glove_path)
    else:
        glove_embeds = pd.read_parquet(f"hf://datasets/Rui98/glove/{glove_name}.parquet")
        glove_embeds.to_parquet(glove_path, index=False)
    return glove_embeds


def load_tokenizer(**kwargs):
    """
    load tokenizer based on the embedding type: glove, plm
    :param kwargs: embedding_type, glove/plm;
    :param kwargs: For glove--subset_name, small/large; use_cached_tokenizer; max_vocab_size; min_token_frequency
    :param kwargs: For plm--embedding_model, e.g. bert-base-uncased;
    :return: Tokenizer object
    """
    embedding_type = kwargs.get("embedding_type", "glove")
    if embedding_type == "glove":
        subset_name = kwargs.get("subset_name", "small")
        use_cached_tokenizer = kwargs.get("use_cached_tokenizer", True)
        default_tokenizer_path = Path(f"{get_project_root()}/cached/{embedding_type}_{subset_name}_tokenizer.json")
        if default_tokenizer_path.exists() and use_cached_tokenizer:
            # load tokenizer from cached file
            from tokenizers import Tokenizer
            tokenizer = Tokenizer.from_file(default_tokenizer_path.as_posix())
        else:
            # train tokenizer from scratch
            from tokenizers import Tokenizer, normalizers
            from tokenizers.trainers import WordLevelTrainer
            from tokenizers.models import WordLevel
            from tokenizers.pre_tokenizers import Whitespace, Punctuation, Digits, Sequence
            from tokenizers.normalizers import Lowercase, NFD, StripAccents
            news_data = load_dataset_from_csv(f"news_{subset_name}")
            news_text = list(news_data["title"]) + list(news_data["abstract"]) + list(news_data["body"])
            tokenizer = Tokenizer(WordLevel(unk_token="[UNK]", vocab=None))
            tokenizer.pre_tokenizer = Sequence([Whitespace(), Punctuation(), Digits()])
            tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
            trainer = WordLevelTrainer(special_tokens=["[UNK]"],
                                       vocab_size=kwargs.get("max_vocab_size", 100000),
                                       min_frequency=kwargs.get("min_token_frequency", 10))
            tokenizer.train_from_iterator(news_text, trainer)

            default_tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
            tokenizer.save(default_tokenizer_path.as_posix())
        # redefine encode function to return ids only
        encode_func = tokenizer.encode
        tokenizer.encode = lambda x: encode_func(x).ids
        tokenizer.pad_token_id = tokenizer.get_vocab()["[UNK]"]
        return tokenizer
    elif embedding_type == "plm":
        from transformers import AutoTokenizer
        embedding_model = kwargs.get("embedding_model", "bert-base-uncased")
        return AutoTokenizer.from_pretrained(embedding_model)
    else:
        raise ValueError("embedding_type must be in ['glove', 'plm']")


def load_glove_embedding_matrix(**kwargs):
    """
    Load embedding matrix from glove embedding
    :param kwargs: subset_name, small/large; use_cached_glove_embed; glove_name
    :return: embedding matrix of numpy array
    """
    glove_name = kwargs.get("glove_name", "glove_6b_300d")
    subset_name = kwargs.get("subset_name", "small")
    glove_embedding_path = Path(f"{get_project_root()}/cached/{glove_name}_{subset_name}.npy")
    use_cached_glove_embed = kwargs.get("use_cached_glove_embed", True)
    if glove_embedding_path.exists() and use_cached_glove_embed:
        # load embedding matrix from cached file
        return np.load(glove_embedding_path.as_posix())
    glove_embeds = load_embedding_from_glove_name(glove_name=glove_name)
    glove_embeds.set_index("token", inplace=True)
    glove_tokenizer = load_tokenizer(embedding_type="glove", subset_name="small")
    glove_vocab = glove_tokenizer.get_vocab()
    vocab_in = set(glove_embeds.index.values) & set(glove_vocab)
    vocab_out = set(glove_vocab) - set(vocab_in)
    vocab_embeds = glove_embeds.loc[list(vocab_in)]
    indices_in = [glove_vocab[token] for token in vocab_embeds.index.values]
    indices_out = [glove_vocab[token] for token in vocab_out]
    embed_vectors = vocab_embeds.values
    # fill out-of-vocabulary news_tokens with random embeddings
    mean, std = np.mean(embed_vectors), np.std(embed_vectors)
    embed_vectors_out = np.random.normal(mean, std, (len(vocab_out), embed_vectors.shape[1]))
    glove_embeddings = np.zeros((len(glove_vocab), embed_vectors.shape[1]))
    glove_embeddings[indices_in] = embed_vectors
    glove_embeddings[indices_out] = embed_vectors_out
    np.save(glove_embedding_path.as_posix(), glove_embeddings)
    return glove_embeddings


def load_entity_embedding_matrix(entity_dict, **kwargs):
    """
    load entity embedding matrix from entity dictionary
    :param entity_dict: entity mapping dictionary
    :param kwargs: subset_name
    :return: entity embedding matrix for the corresponding dictionary
    """
    if "[UNK]" not in entity_dict:
        entity_dict["[UNK]"] = 0
    entity_embedding = load_dataset_from_csv(f"entity_embedding_{kwargs.get('subset_name')}")
    entity_embedding = entity_embedding.to_pandas()
    entity_matrix = np.zeros((len(entity_dict), len(entity_embedding.columns) - 1))
    entity_embedding.set_index("entity", inplace=True)
    missing_rate = 0
    for entity in entity_dict:
        if entity in entity_embedding.index:
            entity_matrix[entity_dict[entity]] = entity_embedding.loc[entity].values
        else:
            missing_rate += 1
    print(f"Missing rate of entity embedding: {round(missing_rate / len(entity_dict), 4)}")
    return entity_matrix


def load_entity_dict(subset_name: str, entity_feature: Union[str, list] = "entity"):
    """
    load entity dictionary from mind dataset
    :param subset_name: subset name of the dataset
    :param entity_feature: entity feature name in the dataset, should be entity or ab_entity
    :return: entity dictionary
    """
    entity_ids = []
    news_data = load_dataset_from_csv(f"news_{subset_name}")
    entity_feature = [entity_feature] if isinstance(entity_feature, str) else entity_feature
    for feature in entity_feature:
        for entity in list(news_data[feature]):
            if entity:
                entity_dict = json.loads(entity)
                entity_ids.extend([obj["WikidataId"] for obj in entity_dict])
    entity_ids = ["[UNK]"] + list(set(entity_ids))
    return dict(zip(entity_ids, range(0, len(entity_ids))))


class FeatureMapper:

    def get_entity_matrix(self):
        start = self.title_length + self.abstract_length + self.body_length + 2
        return self.feature_matrix[..., start:]

    def __init__(self, **kwargs):
        self.subset_name = kwargs.get("subset_name", "small")
        self.title_length = kwargs.get("title_len", 30)
        self.abstract_length = kwargs.get("abstract_len", 30)
        self.body_length = kwargs.get("body_len", 100)
        self.tokenizer = load_tokenizer(**kwargs)
        self.entity_feature = kwargs.get("entity_feature")  # entity: title entity; ab_entity: plus abstract
        # feature include: title, abstract, body, category(1), subcategory(1)
        self.feature_dim = self.title_length + self.abstract_length + self.body_length + 2
        if self.entity_feature is not None:
            self.entity_length = kwargs.get("entity_len", 5)
            self.entity_feature = [self.entity_feature] if isinstance(self.entity_feature, str) else self.entity_feature
            self.entity_dict = load_entity_dict(self.subset_name, self.entity_feature)
            self.feature_dim += (self.entity_length * len(self.entity_feature))
        news_data = load_dataset_from_csv(f"news_{self.subset_name}")
        self.feature_matrix = np.zeros((len(news_data) + 1, self.feature_dim), dtype=np.int32)
        category, subvert, nid = list(news_data["category"]), list(news_data["subvert"]), list(news_data["nid"])
        self.category_mapper = {c: i + 1 for i, c in enumerate(set(category))}
        self.subvert_mapper = {s: i + 1 for i, s in enumerate(set(subvert))}
        title, abstract, body = list(news_data["title"]), list(news_data["abstract"]), list(news_data["body"])
        entity_lists = {"entity": list(news_data["entity"]), "ab_entity": list(news_data["ab_entity"])}
        for index in range(len(news_data)):
            title_tokens = self.tokenizer.encode(title[index])[:self.title_length]
            title_tokens += [0] * (self.title_length - len(title_tokens))
            abstract_tokens = self.tokenizer.encode(abstract[index])[:self.abstract_length]
            abstract_tokens += [0] * (self.abstract_length - len(abstract_tokens))
            body_tokens = self.tokenizer.encode(body[index])[:self.body_length]
            body_tokens += [0] * (self.body_length - len(body_tokens))
            category_id = self.category_mapper[category[index]]
            subvert_id = self.subvert_mapper[subvert[index]]
            data = [title_tokens, abstract_tokens, body_tokens, [category_id, subvert_id]]
            if self.entity_feature is not None:
                for e_f in self.entity_feature:
                    entity_line = entity_lists[e_f][index]
                    if entity_line and len(entity_line):
                        entity_tokens = [self.entity_dict[obj["WikidataId"]] for obj in json.loads(entity_line)]
                    else:
                        entity_tokens = [self.entity_dict["[UNK]"]]
                    entity_tokens = entity_tokens[:self.entity_length]
                    entity_tokens += [0] * (self.entity_length - len(entity_tokens))
                    data.append(entity_tokens)
            self.feature_matrix[nid[index]] = np.concatenate(data)


def load_feature_mapper(**kwargs):
    """
    Load feature mapper from cached file or create a new one
    :param kwargs: subset_name, small/large; use_cached_feature_mapper
    :return: FeatureMapper object
    """
    use_cached_feature_mapper = kwargs.get("use_cached_feature_mapper", True)
    title_len, subset_name = kwargs.get("title_len", 30), kwargs.get('subset_name', 'small')
    abstract_len, body_len = kwargs.get("abstract_len", 30), kwargs.get("body_len", 100)
    embed_dim = title_len + abstract_len + body_len + 2
    if kwargs.get("entity_len"):
        entity_feature = kwargs.get("entity_feature")
        entity_feature = [entity_feature] if isinstance(entity_feature, str) else entity_feature
        embed_dim += (kwargs.get("entity_len") * len(entity_feature))
    tokenizer_name = kwargs.get("embedding_type", "glove")
    if tokenizer_name == "plm":  # pre-trained LM: use corresponding embedding model name
        tokenizer_name = kwargs.get("embedding_model")
    feature_mapper_path = Path(f"{get_project_root()}/cached/FM_{tokenizer_name}_{embed_dim}d_{subset_name}.bin")
    feature_mapper = None
    if feature_mapper_path.exists() and use_cached_feature_mapper:
        with open(feature_mapper_path, "rb") as f:
            try:
                feature_mapper = pickle.load(f)
            except ModuleNotFoundError:
                pass
    if feature_mapper is None:
        feature_mapper = FeatureMapper(**kwargs)
        with open(feature_mapper_path, "wb") as f:
            pickle.dump(feature_mapper, f)
    return feature_mapper


def load_user_history_mapper(**kwargs):
    """
    Load user history mapper from cached file or create a new one
    :param kwargs: subset_name, max_history_size
    :return: user_history_mapper with size (num_users, max_history_size)
    """
    subset_name = kwargs.get("subset_name")
    max_history_size = kwargs.get("max_history_size")
    if subset_name is None or max_history_size is None:
        raise ValueError("subset_name and max_history_size must be provided")
    user_interaction = load_dataset_from_csv(f"user_interaction_{subset_name}")
    user_history_mapper = np.zeros((len(user_interaction) + 1, max_history_size), dtype=np.int32)
    # fetch uid and history to two lists
    history_nid, history_uid = list(user_interaction["history"]), list(user_interaction["uid"])
    for index in range(len(user_interaction)):
        history = history_nid[index]
        history = history.split() if history else [0]
        history = history[-max_history_size:] + [0] * (max_history_size - len(history))
        user_history_mapper[history_uid[index]] = np.asarray(history, dtype=np.int32)
    return user_history_mapper
