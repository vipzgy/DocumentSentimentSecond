# -*- encoding: utf-8 -*-
import re
import codecs
import _pickle
from collections import Counter


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
        # self.predict_sentiment = None


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
        # self.predict_labels = []


def read_doc(doc_path, s_path, max_length, is_train=False):
    paragraphs_data = []
    sentence_length_counter = Counter()
    word_counter = Counter()
    para_label_counter = Counter()
    s_label_counter = Counter()
    with open(doc_path, 'r', encoding='utf-8') as para_file, open(s_path, 'r', encoding='utf-8') as s_file:
        for para_line in para_file.readlines():
            para_line = para_line.strip()
            para_strs = para_line.split('\t')
            t_label = para_strs[-1]
            if t_label == '?':
                t_label = '0'
            if t_label == '2':
                t_label = '1'
            paragraph = Paragraph(t_label)
            para_label_counter[t_label] += 1
            sentences_number = int(para_strs[1])

            # sentence
            sentences_count = 0
            sentence = []
            starts = []
            ends = []
            labels = []
            word_count = 0
            target_flag = False
            for s_line in s_file:
                s_line = s_line.strip()
                if len(s_line) == 0 or s_line == '':
                    if len(sentence) > max_length:
                        sentences_number -= 1
                    else:
                        if len(starts) == len(ends) == len(labels):
                            paragraph.sentences.append(Sentence(sentence, starts, ends, labels))
                            sentence_length_counter[len(sentence)] += 1
                            sentences_count += 1
                        else:
                            print("target starts!=ends!=s_labels")
                            print(sentence)
                            exit()
                    sentence = []
                    starts = []
                    ends = []
                    labels = []
                    word_count = 0
                    if sentences_count == sentences_number:
                        paragraphs_data.append(paragraph)
                        break
                else:
                    strings = s_line.split('\t')
                    sentence.append(strings[0])
                    word_count += 1
                    if is_train:
                        word_counter[strings[0]] += 1
                    if strings[2] == 'e':
                        if target_flag:
                            ends.append(word_count - 1)
                            target_flag = False
                        else:
                            print("something wrong.")
                    elif strings[2] == 'b':
                        starts.append(word_count - 1)
                        labels.append(strings[3])
                        target_flag = True
                        if is_train:
                            s_label_counter[strings[3] if strings[3] != '?' else '0'] += 1
                    elif strings[2] == 's':
                        starts.append(word_count - 1)
                        ends.append(word_count - 1)
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
                paragraph.sentences.append(Sentence(sentence, starts, ends, labels))
                sentence_length_counter[len(sentence)] += 1
                if sentences_count == sentences_number:
                    paragraphs_data.append(paragraph)
    if is_train:
        return paragraphs_data, sentence_length_counter, word_counter, para_label_counter, s_label_counter
    return paragraphs_data, sentence_length_counter


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
