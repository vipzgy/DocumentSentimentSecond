# -*- coding: utf-8 -*-
import os
import torch
import numpy
import random
import torch.nn.functional as F
from target.TorchNN import *
from target.driver.Config import Configurable
from target.driver.MyIO import read_pkl
from target.driver.Vocab import PAD, VocabTgt, VocabSrc
from target.driver.Train import predict

# 获取当前路径
sentence_path = os.path.split(os.path.realpath(__file__))[0]
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["OMP_NUM_THREADS"] = "1"


class CompanyPredict:
    def __init__(self):
        # random
        torch.manual_seed(666)
        random.seed(666)
        numpy.random.seed(666)

        # gpu
        gpu = torch.cuda.is_available()
        print("GPU available: ", gpu)
        print("CuDNN: ", torch.backends.cudnn.enabled)
        if gpu:
            torch.cuda.manual_seed(666)

        # parameters
        config_file = os.path.join(sentence_path, 'default.ini')
        thread = 1
        use_cuda = False
        model = 'model.64190'

        self.config = Configurable(config_file, [])
        torch.set_num_threads(thread)
        self.config.use_cuda = False
        if gpu and use_cuda:
            self.config.use_cuda = True
        # print("\nGPU using status: ", self.config.use_cuda)

        # load vocab and model
        self.feature_list = read_pkl(os.path.join(sentence_path, self.config.load_feature_voc))
        self.label_list = read_pkl(os.path.join(sentence_path, self.config.load_label_voc))
        self.feature_vec = VocabSrc(self.feature_list)
        self.label_vec = VocabTgt(self.label_list)

        # model
        if self.config.which_model == 'Vanilla':
            self.model = Vanilla(self.config, self.feature_vec.size, self.config.embed_dim,
                                 PAD, self.label_vec.size)
        elif self.config.which_model == 'Contextualized':
            self.model = Contextualized(self.config, self.feature_vec.size, self.config.embed_dim,
                                        PAD, self.label_vec.size)
        elif self.config.which_model == 'ContextualizedGates':
            self.model = ContextualizedGates(self.config, self.feature_vec.size, self.config.embed_dim,
                                             PAD, self.label_vec.size)
        else:
            print('please choose right model')
            exit()
        model_path = os.path.join(self.config.load_model_path, model)
        self.model.load_state_dict(torch.load(os.path.join(sentence_path, model_path),
                                              map_location=lambda storage, loc: storage))
        if self.config.use_cuda:
            torch.backends.cudnn.enabled = True
            self.model = model.cuda()

    def predict(self, para):
        for idx in range(len(para.sentences)):
            for idj in range(len(para.sentences[idx].starts)):
                data = (para.sentences[idx].words, para.sentences[idx].starts[idj], para.sentences[idx].ends[idj])
                logit = predict(self.model, data, self.feature_vec, self.config)
                logit = F.softmax(logit, dim=1)
                tem = {}
                for i in range(len(self.label_list)):
                    tem[self.label_vec.id2word(i)] = round(logit.data[0][i], 5)
                # 转换成为三分类的概率值
                # result = {}
                # result['positive'] = tem['1'] + tem['2']
                # result['negative'] = tem['-1'] + tem['-2']
                # result['neutral'] = tem['0']
                # para.sentences[idx].targets[idj].sentiment = result
                para.sentences[idx].predict_labels.append(tem)
        return para.sentences

