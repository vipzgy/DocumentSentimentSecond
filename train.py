# -*- coding: utf-8 -*-
import os
import torch
import numpy
import random
import argparse
from TorchNN import *
from driver.Config import Configurable
from driver.MyIO import read_pkl
from driver.Vocab import PAD, VocabSrc, VocabTgt
from driver.Train import train

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"


class Paragraph(object):
    def __init__(self):
        self.sentences = []
        # paragraph sentiment
        self.sentiment = None


class Sentence(object):
    def __init__(self):
        self.words = []
        self.targets = []


class Target(object):
    def __init__(self):
        self.start = -1
        self.end = -1
        # target sentiment
        self.sentiment = None


if __name__ == '__main__':
    # random
    torch.manual_seed(666)
    random.seed(666)
    numpy.random.seed(666)

    gpu = torch.cuda.is_available()
    if gpu:
        torch.cuda.manual_seed(666)
    print("GPU available: ", gpu)
    print("CuDNN: ", torch.backends.cudnn.enabled)

    # parameters
    parse = argparse.ArgumentParser('Attention Target Classifier')
    parse.add_argument('--config_file', type=str, default='default.ini')
    parse.add_argument('--thread', type=int, default=1)
    parse.add_argument('--use_cuda', action='store_true', default=True)
    args, extra_args = parse.parse_known_args()

    config = Configurable(args.config_file, extra_args)
    torch.set_num_threads(args.thread)
    config.use_cuda = False
    if gpu and args.use_cuda:
        config.use_cuda = True
    print("\nGPU using status: ", config.use_cuda)

    # load data
    train_data = read_pkl(config.train_pkl)
    dev_data = None
    if config.para_dev_file:
        dev_data = read_pkl(config.dev_pkl)
    test_data = read_pkl(config.test_pkl)

    word_list = read_pkl(config.load_feature_voc)
    p_label_list, s_label_list = read_pkl(config.load_label_voc)
    word_voc = VocabSrc(word_list)
    p_label_voc = VocabTgt(p_label_list)
    s_label_voc = VocabTgt(s_label_list)

    embedding = None
    if os.path.isfile(config.embedding_pkl):
        embedding = read_pkl(config.embedding_pkl)

    # model
    model = None
    if config.which_model == 'MyParaCNN':
        model = MyParaCNN(config, word_voc.size, embedding[1] if embedding else config.embedding_dim,
                          PAD, p_label_voc.size, embedding[0] if embedding else None)
    elif config.which_model == 'MyParaLSTM':
        model = MyParaLSTM(config, word_voc.size, embedding[1] if embedding else config.embedding_dim,
                           PAD, p_label_voc.size, embedding[0] if embedding else None)
    elif config.which_model == 'HierarchicalAttentionLSTM':
        model = HierarchicalAttentionLSTM(config, word_voc.size,
                                          embedding[1] if embedding else config.embedding_dim,
                                          PAD, p_label_voc.size, embedding[0] if embedding else None)
    else:
        print('please choose right model')
        exit()

    if config.use_cuda:
        torch.backends.cudnn.enabled = True
        model = model.cuda()

    # train
    train(model, train_data, dev_data, test_data, word_voc, p_label_voc, s_label_voc, config)

