# -*- coding: utf-8 -*-
# @Author        : Rui
# @Time          : 2024/5/6 20:25
# @Function      :
import omegaconf


class Configuration:
    def __init__(self, **kwargs):
        self.config = omegaconf.OmegaConf.create(kwargs)

    def __getitem__(self, item):
        return self.config[item]

    def __setitem__(self, key, value):
        self.config[key] = value

    def __getattr__(self, item):
        return self.config[item]

    def __setattr__(self, key, value):
        self.config[key] = value

    def __str__(self):
        return omegaconf.OmegaConf.to_yaml(self.config)

    def __repr__(self):
        return omegaconf.OmegaConf.to_yaml(self.config)

    def save(self, path):
        omegaconf.OmegaConf.save(self.config, path)

    def load(self, path):
        self.config = omegaconf.OmegaConf.load(path)

    def merge_with_dotlist(self, dotlist):
        self.config = omegaconf.OmegaConf.merge(self.config, omegaconf.OmegaConf.from_dotlist(dotlist))

    def merge_with_dict(self, dict_config):
        self.config = omegaconf.OmegaConf.merge(self.config, omegaconf.OmegaConf.create(dict_config))

    def merge_with_file(self, file_path):
        self.config = omegaconf.OmegaConf.merge(self.config, omegaconf.OmegaConf.load(file_path))

    def merge_with_cli(self):
        self.config = omegaconf.OmegaConf.from_cli().merge(self.config)
