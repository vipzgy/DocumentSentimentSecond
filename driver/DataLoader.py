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


def pair_data_variable(data, vocab_srcs, vocab_p, config):
    sentences_batchs = create_batch_iter(data.sentences, config.batch_size, shuffle=True)

    srcs = []
    tgts = []
    src_lens = []
    para_tgt = data.sentiment

    para_words = Variable(torch.LongTensor(1).zero_())
    para_words.data[0] = vocab_p.word2id(para_tgt)

    for batch in sentences_batchs:
        batch_size = len(batch)
        batch = sorted(batch, key=lambda b: len(b.words), reverse=True)
        src_lengths = [len(batch[i].words) for i in range(batch_size)]
        max_src_length = int(src_lengths[0])

        src_words = Variable(torch.LongTensor(max_src_length, batch_size).zero_(), requires_grad=False)
        tgt_labels = Variable(torch.LongTensor(max_src_length, batch_size).zero_(), requires_grad=False)
        # 没有必要记录target的位置，因为现在的做法是直接cat后平均

        for idx, instance in enumerate(batch):
            sentence = vocab_srcs.word2id(instance.words)
            for index, word in enumerate(sentence):
                src_words.data[index][idx] = word
            for id_label, (id_start, id_end) in enumerate(zip(instance.starts, instance.ends)):
                for id in range(id_start, id_end + 1):
                    label = instance.labels[id_label]
                    if label == 's_positive':
                        label = 'positive'
                    elif label == 's_negative':
                        label = 'negative'
                    else:
                        pass
                    tgt_labels.data[id][idx] = vocab_srcs.word2id(label)

        if config.use_cuda:
            src_words = src_words.cuda()
            tgt_labels = tgt_labels.cuda()
            para_words = para_words.cuda()
        srcs.append(src_words)
        tgts.append(tgt_labels)
        src_lens.append(src_lengths)

    return srcs, tgts, para_words, src_lens


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
