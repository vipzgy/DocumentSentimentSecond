# -*- encoding: utf-8 -*-
import re
import codecs
import _pickle
from collections import Counter


def read_word_line(sentence_path, config, is_train=False):
    data = []
    sentence_len = Counter()
    word_dict = Counter()
    label_dict = Counter()
    with open(sentence_path, 'r', encoding='utf-8') as sentence_file:
        sentence = []
        starts = []
        ends = []
        labels = []
        count = 0
        target_flag = False
        for s_line in sentence_file:
            s_line = s_line.strip()
            if len(s_line) == 0 or s_line == '':
                if len(sentence) > config.max_length:
                    pass
                else:
                    if len(starts) == len(ends) == len(labels):
                        for i in range(len(labels)):
                            data.append((sentence, starts[i], ends[i], labels[i]))
                            sentence_len[len(sentence)] += 1
                            if is_train:
                                label_dict[labels[i]] += 1
                    else:
                        print(sentence)
                        exit()
                sentence = []
                starts = []
                ends = []
                labels = []
                count = 0
            else:
                strings = s_line.split('\t')
                sentence.append(strings[0])
                count += 1
                if is_train:
                    word_dict[strings[0]] += 1
                if strings[2] == 'e':
                    if target_flag:
                        ends.append(count - 1)
                        target_flag = False
                    else:
                        print("something wrong.")
                elif strings[2] == 'b':
                    starts.append(count - 1)
                    labels.append(strings[3])
                    target_flag = True
                elif strings[2] == 's':
                    starts.append(count - 1)
                    ends.append(count - 1)
                    labels.append(strings[3])
                elif strings[2] == 'o':
                    if target_flag is True:
                        print("something wrong.")
                elif strings[2] == 'm':
                    if target_flag is False:
                        print("something wrong.")
                else:
                    print("something wrong.")
        if len(sentence) != 0:
            if len(sentence) > config.max_length:
                pass
            else:
                if len(starts) == len(ends) == len(labels):
                    for i in range(len(labels)):
                        data.append((sentence, starts[i], ends[i], labels[i]))
                        sentence_len[len(sentence)] += 1
                else:
                    print(sentence)
                    exit()
    if is_train:
        return data, sentence_len, word_dict, label_dict
    return data, sentence_len


def read_pkl(path):
    """
    读取pkl文件

    Args:
        path: str, pkl文件路径

    Return:
        pkl_ob: pkl对象
    """
    file_pkl = codecs.open(path, 'rb')
    return _pickle.load(file_pkl)
