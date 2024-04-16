# -*- coding: utf-8 -*-
# @Author        : Rui
# @Time          : 2024/4/14 15:25
# @Function      : Define the utils functions processing news data
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
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


class FeatureMapper:
    def __init__(self, **kwargs):
        self.subset_name = kwargs.get("subset_name", "small")
        self.title_length = kwargs.get("title_len", 30)
        self.abstract_length = kwargs.get("abstract_len", 30)
        self.body_length = kwargs.get("body_len", 100)
        self.tokenizer = load_tokenizer(**kwargs)
        # feature include: title, abstract, body, category(1), subcategory(1)
        self.feature_dim = self.title_length + self.abstract_length + self.body_length + 2
        news_data = load_dataset_from_csv(f"news_{self.subset_name}")
        self.feature_matrix = np.zeros((len(news_data) + 1, self.feature_dim), dtype=np.int32)
        category, subvert, nid = list(news_data["category"]), list(news_data["subvert"]), list(news_data["nid"])
        self.category_mapper = {c: i + 1 for i, c in enumerate(set(category))}
        self.subvert_mapper = {s: i + 1 for i, s in enumerate(set(subvert))}
        title, abstract, body = list(news_data["title"]), list(news_data["abstract"]), list(news_data["body"])
        for index in range(len(news_data)):
            title_tokens = self.tokenizer.encode(title[index])[:self.title_length]
            title_tokens += [0] * (self.title_length - len(title_tokens))
            abstract_tokens = self.tokenizer.encode(abstract[index])[:self.abstract_length]
            abstract_tokens += [0] * (self.abstract_length - len(abstract_tokens))
            body_tokens = self.tokenizer.encode(body[index])[:self.body_length]
            body_tokens += [0] * (self.body_length - len(body_tokens))
            category_id = self.category_mapper[category[index]]
            subvert_id = self.subvert_mapper[subvert[index]]
            self.feature_matrix[nid[index]] = np.concatenate(
                [title_tokens, abstract_tokens, body_tokens, [category_id, subvert_id]]
            )


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
    feature_mapper_path = Path(f"{get_project_root()}/cached/feature_{embed_dim}d_{subset_name}.bin")
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
