# -*- coding: utf-8 -*-
import os
import time
from target.company_classifier import CompanyPredict
from driver.Config import Configurable
from driver.MyIO import read_pkl
from driver.Vocab import VocabSrc, VocabTgt


class Paragraph:
    """
        存储句子,一个句子可能有很多个实体

        Parameters:
            self.sentences: list 存储所有的句子
            self.sentiment: int存储段落的情感

        """
    def __init__(self, sentiment):
        self.sentences = []
        self.sentiment = sentiment
        self.predict_sentiment = None


class Sentence:
    """
        存储句子,一个句子可能有很多个实体

        Parameters:
            self.words: list
            self.starts: list
            self.ends: list
            self.labels: list

    """
    def __init__(self, words, starts, ends, labels):
        self.words = words
        self.starts = starts
        self.ends = ends
        self.labels = labels
        self.predict_labels = []


if __name__ == '__main__':
    # parameters
    config_file = 'default.ini'
    config = Configurable(config_file)

    # model
    model = CompanyPredict()
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

    # 开始
    corrects, size = 0, 0
    start_time = time.time()
    # 初始化宏平均
    macro_averaging = {}
    for i in p_label_voc.i2w:
        macro_averaging[i] = {'tp': 0, 'fn': 0, 'fp': 0}

    for idx, data in enumerate(test_data):
        if len(data.sentences) == 0:
            continue
        # 获得多有target的情感极性
        model.predict(data)
        # 根据所有target的平均值来计算段落的情感极性
        positive, neutral, negative = 0.0, 0.0, 0.0
        count = 0
        for idx in range(len(data.sentences)):
            for idj in range(len(data.sentences[idx].predict_labels)):
                count += 1
                positive += data.sentences[idx].predict_labels[idj]['1'] + data.sentences[idx].predict_labels[idj]['2']
                neutral += data.sentences[idx].predict_labels[idj]['0']
                negative += data.sentences[idx].predict_labels[idj]['-1'] + data.sentences[idx].predict_labels[idj]['-2']

        positive = positive/count
        neutral = neutral/count
        negative = negative/count
        if (positive + neutral + negative) != 1.0:
            print('------这种计算方法不是太好')
        if positive > neutral:
            if positive > negative:
                logit = '1'
        else:
            if neutral > negative:
                logit = '0'
            else:
                logit = '-1'
        if logit == data.sentiment:
            corrects += 1
        size += 1

        # 统计宏平均
        if logit == data.sentiment:
            macro_averaging[data.sentiment]['tp'] += 1
        else:
            macro_averaging[data.sentiment]['fn'] += 1
            macro_averaging[logit]['fp'] += 1
    # 计算宏平均
    macro_precesion = 0
    macro_recall = 0
    count = 0
    for k in macro_averaging.keys():
        tp_fp = macro_averaging[k]['tp'] + macro_averaging[k]['fp']
        tp_fn = macro_averaging[k]['tp'] + macro_averaging[k]['fn']
        if tp_fp == 0:
            # 就是是？的时候，这个都不计算在内
            # 如果要计算，就是count += 1
            count += 1
        else:
            macro_precesion += macro_averaging[k]['tp'] / tp_fp
            macro_recall += macro_averaging[k]['tp'] / tp_fn
            count += 1
    macro_precesion = macro_precesion / count
    macro_recall = macro_recall / count
    macro_averaging_result = 2 * macro_precesion * macro_recall / (macro_precesion + macro_recall)

    accuracy = 100.0 * corrects / size
    during_time = float(time.time() - start_time)
    print("\nevaluate result: ")
    print("accuracy:{:.4f}({}/{}), macro_averaging:{:.4f}, time:{:.2f}"
          .format(accuracy, corrects, size, macro_averaging_result, during_time))
