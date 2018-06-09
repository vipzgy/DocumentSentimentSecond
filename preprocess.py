# -*- coding: utf-8 -*-
import os
import numpy
import pickle
import random
import argparse
from driver.Config import Configurable

from driver.Vocab import VocabSrc, VocabTgt


def analysis(data, sentence_length, word_dict=None, para_label_dict=None, s_label_dict=None):

    if word_dict is not None:
        print('单词个数为: ', len(word_dict))
    if para_label_dict is not None:
        print('段落的标签个数有: {0}个'.format(len(para_label_dict)))
        print('标签有: ')
        for k in para_label_dict.keys():
            print("标签为：{0}， 个数为：{1}".format(k, para_label_dict[k]))
    if s_label_dict is not None:
        print('句子的标签个数有: {0}个'.format(len(s_label_dict)))
        print('标签有: ')
        for k in s_label_dict.keys():
            print("标签为：{0}， 个数为：{1}".format(k, s_label_dict[k]))

    # 每段包含几句话
    print("每段中包含的句子数")
    max_sentence_number = -1
    for idx, d in enumerate(data):
        if len(d.sentences) > max_sentence_number:
            max_sentence_number = len(d.sentences)
        print(str(idx), str(len(d.sentences)))
    print("最大个数为：", str(max_sentence_number))

    sentence_length = sorted(sentence_length.items(), key=lambda item: item[0], reverse=False)
    # sentence_length = sentence_length.most_common()
    count = 0
    for item in sentence_length:
        print("句子长度为： {0}  有{1}句".format(item[0], item[1]))
        count += int(item[1])
    print('句子个数为：', count)


if __name__ == '__main__':
    # random
    random.seed(666)
    numpy.random.seed(666)

    # parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default='default.ini')
    parser.add_argument('--thread', type=int, default=1)
    args, extra_args = parser.parse_known_args()
    config = Configurable(args.config_file, extra_args)

    # data and analysis
    from driver.MyIO import read_doc

    print('\n')
    train_data, train_sentence_len, word_dict, para_label_dict, s_label_dict = read_doc(
        config.para_train_file, config.sen_train_file, config.max_length, is_train=True)
    analysis(train_data, train_sentence_len, word_dict, para_label_dict, s_label_dict)
    # some corpus do not have dev data set
    if config.para_dev_file:
        print('\n')
        dev_data, dev_sentence_len = read_doc(config.para_dev_file, config.sen_dev_file, config.max_length)
        analysis(dev_data, dev_sentence_len)
    print('\n')
    test_data, test_sentence_len = read_doc(config.para_test_file, config.sen_test_file, config.max_length)
    analysis(test_data, test_sentence_len)

    if not os.path.isdir(config.save_dir):
        os.mkdir(config.save_dir)
    pickle.dump(train_data, open(config.train_pkl, 'wb'))
    if config.para_dev_file:
        pickle.dump(dev_data, open(config.dev_pkl, 'wb'))
    pickle.dump(test_data, open(config.test_pkl, 'wb'))

    # vocab
    feature_list = [k for k, v in word_dict.most_common(config.vocab_size)]
    para_label_list = [k for k in para_label_dict.keys()]
    s_label_list = [k for k in s_label_dict.keys()]
    pickle.dump(feature_list, open(config.save_feature_voc, 'wb'))
    pickle.dump((para_label_list, s_label_list), open(config.save_label_voc, 'wb'))

    feature_voc = VocabSrc(feature_list)
    p_label_voc = VocabTgt(para_label_dict)
    s_label_voc = VocabTgt(s_label_dict)

    # embedding
    if config.embedding_file:
        embedding = feature_voc.create_vocab_embs(config.embedding_file)
        pickle.dump(embedding, open(config.embedding_pkl, 'wb'))


