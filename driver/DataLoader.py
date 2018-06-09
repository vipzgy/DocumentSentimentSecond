# -*- coding: utf-8 -*-
import torch
import numpy as np
from torch.autograd import Variable


def create_batch_iter(data, batch_size, shuffle=False):
    if shuffle:
        np.random.shuffle(data)

    batched_data = []
    instances = []
    count = 0
    for instance in data:
        instances.append(instance)
        count += 1
        if count == batch_size:
            batched_data.append(instances)
            instances = []
            count = 0
    if count != 0:
        batched_data.append(instances)
    return batched_data


def pair_data_variable(data, vocab_srcs, vocab_tgts, config):
    sentences_batchs = create_batch_iter(data.sentences, config.batch_size, shuffle=True)

    srcs = []
    para_tgt = data.sentiment
    # tgts = []
    # starts = []
    # ends = []
    src_lens = []

    para_words = Variable(torch.LongTensor(1).zero_())
    para_words.data[0] = vocab_tgts.word2id(para_tgt)

    for batch in sentences_batchs:
        batch_size = len(batch)
        batch = sorted(batch, key=lambda b: len(b.words), reverse=True)
        src_lengths = [len(batch[i].words) for i in range(batch_size)]
        max_src_length = int(src_lengths[0])

        src_words = Variable(torch.LongTensor(max_src_length, batch_size).zero_(), requires_grad=False)
        # tgt_words = Variable(torch.LongTensor(batch_size).zero_(), requires_grad=False)

        # start = []
        # end = []

        for idx, instance in enumerate(batch):
            sentence = vocab_srcs.word2id(instance.words)
            for index, word in enumerate(sentence):
                src_words.data[index][idx] = word
            # tgt_words.data[idx] = vocab_tgts.word2id(instance[3])
            # start.append(instance[1])
            # end.append(instance[2])

        if config.use_cuda:
            src_words = src_words.cuda()
            # tgt_words = tgt_words.cuda()
            para_words = para_words.cuda()
        srcs.append(src_words)
        # tgts.append(tgt_words)
        # starts.append(start)
        # ends.append(end)
        src_lens.append(src_lengths)

    # return srcs, tgts, starts, ends, src_lens
    return srcs, para_words, src_lens


def pair_data_variable_predict(data, vocab_srcs, config):

    src_words = Variable(torch.LongTensor(len(data[0]), 1).zero_(), requires_grad=False)

    start = []
    end = []
    sentence = vocab_srcs.word2id(data[0])
    for index, word in enumerate(sentence):
        src_words.data[index][0] = word
    start.append(data[1])
    end.append(data[2])

    if config.use_cuda:
        src_words = src_words.cuda()

    return src_words, start, end, [len(data[0])]
