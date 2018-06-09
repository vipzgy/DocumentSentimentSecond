# -*- coding: utf-8 -*-
import os
from configparser import ConfigParser


class Configurable(object):
    def __init__(self, config_file, extra_args):
        config = ConfigParser()
        config.read(config_file)
        if extra_args:
            extra_args = dict([(k[2:], v) for k, v in zip(extra_args[0::2], extra_args[1::2])])
        for section in config.sections():
            for k, v in config.items(section):
                if k in extra_args:
                    v = type(v)(extra_args[k])
                    config.set(section, k, v)

        self._config = config
        if not os.path.isdir(self.save_model_path):
            os.mkdir(self.save_model_path)
        config.write(open(self.config_file, 'w'))
        print('Load config file successfully.\n')
        for section in config.sections():
            for k, v in config.items(section):
                print(k, v)

# Data
    @property
    def data_dir(self):
        return self._config.get('Data', 'data_dir')

    @property
    def para_train_file(self):
        return self._config.get('Data', 'para_train_file')

    @property
    def sen_train_file(self):
        return self._config.get('Data', 'sen_train_file')

    @property
    def para_dev_file(self):
        return self._config.get('Data', 'para_dev_file')

    @property
    def sen_dev_file(self):
        return self._config.get('Data', 'sen_dev_file')

    @property
    def para_test_file(self):
        return self._config.get('Data', 'para_test_file')

    @property
    def sen_test_file(self):
        return self._config.get('Data', 'sen_test_file')

    @property
    def embedding_file(self):
        return self._config.get('Data', 'embedding_file')

    @property
    def vocab_size(self):
        return self._config.getint('Data', 'vocab_size')

    @property
    def max_length(self):
        return self._config.getint('Data', 'max_length')

    @property
    def shuffle(self):
        return self._config.getboolean('Data', 'shuffle')

# Save
    @property
    def save_dir(self):
        return self._config.get('Save', 'save_dir')

    @property
    def config_file(self):
        return self._config.get('Save', 'config_file')

    @property
    def save_model_path(self):
        return self._config.get('Save', 'save_model_path')

    @property
    def save_feature_voc(self):
        return self._config.get('Save', 'save_feature_voc')

    @property
    def save_label_voc(self):
        return self._config.get('Save', 'save_label_voc')

    @property
    def load_dir(self):
        return self._config.get('Save', 'load_dir')

    @property
    def load_model_path(self):
        return self._config.get('Save', 'load_model_path')

    @property
    def load_feature_voc(self):
        return self._config.get('Save', 'load_feature_voc')

    @property
    def load_label_voc(self):
        return self._config.get('Save', 'load_label_voc')

    @property
    def train_pkl(self):
        return self._config.get('Save', 'train_pkl')

    @property
    def dev_pkl(self):
        return self._config.get('Save', 'dev_pkl')

    @property
    def test_pkl(self):
        return self._config.get('Save', 'test_pkl')

    @property
    def embedding_pkl(self):
        return self._config.get('Save', 'embedding_pkl')

# Network
    @property
    def embedding_dim(self):
        return self._config.getint('Network', 'embedding_dim')

    @property
    def dropout_embed(self):
        return self._config.getfloat('Network', 'dropout_embed')

    @property
    def s_num_layers(self):
        return self._config.getint('Network', 's_num_layers')

    @property
    def s_hidden_size(self):
        return self._config.getint('Network', 's_hidden_size')

    @property
    def s_dropout_rnn(self):
        return self._config.getfloat('Network', 's_dropout_rnn')

    @property
    def p_num_layers(self):
        return self._config.getint('Network', 'p_num_layers')

    @property
    def p_hidden_size(self):
        return self._config.getint('Network', 'p_hidden_size')

    @property
    def p_dropout_rnn(self):
        return self._config.getfloat('Network', 'p_dropout_rnn')

    @property
    def s_attention_size(self):
        return self._config.getint('Network', 's_attention_size')

    @property
    def p_attention_size(self):
        return self._config.getint('Network', 'p_attention_size')

    @property
    def in_channels(self):
        return self._config.getint('Network', 'in_channels')

    @property
    def out_channels(self):
        return self._config.getint('Network', 'out_channels')

    @property
    def kernel_sizes(self):
        return self._config.get('Network', 'kernel_sizes')

    @property
    def max_norm(self):
        return self._config.getfloat('Network', 'max_norm')

    @property
    def which_model(self):
        return self._config.get('Network', 'which_model')

# Optimizer
    @property
    def learning_algorithm(self):
        return self._config.get('Optimizer', 'learning_algorithm')

    @property
    def lr(self):
        return self._config.getfloat('Optimizer', 'lr')

    @property
    def lr_scheduler(self):
        return self._config.get('Optimizer', 'lr_scheduler')

    @property
    def weight_decay(self):
        return self._config.getfloat('Optimizer', 'weight_decay')

    @property
    def clip_norm(self):
        return self._config.getfloat('Optimizer', 'clip_norm')

# Run
    @property
    def epochs(self):
        return self._config.getint('Run', 'epochs')

    @property
    def batch_size(self):
        return self._config.getint('Run', 'batch_size')

    @property
    def test_interval(self):
        return self._config.getint('Run', 'test_interval')

    @property
    def save_after(self):
        return self._config.getint('Run', 'save_after')
