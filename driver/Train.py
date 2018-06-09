# -*- coding: utf-8 -*-
import os
import time
import torch
import random
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from driver.DataLoader import create_batch_iter, pair_data_variable, pair_data_variable_predict


def train(model, train_data, dev_data, test_data, vocab_srcs, vocab_para, vocab_s, config):
    # optimizer
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    if config.learning_algorithm == 'sgd':
        optimizer = optim.SGD(parameters, lr=config.lr, weight_decay=config.weight_decay)
    elif config.learning_algorithm == 'adam':
        optimizer = optim.Adam(parameters, lr=config.lr, weight_decay=config.weight_decay)
    else:
        raise RuntimeError('Invalid optimizer method: ' + config.learning_algorithm)

    # train
    global_step = 0
    best_acc = 0
    best_macro = 0
    best_micro = 0
    print('\nstart training...')
    for iter in range(config.epochs):
        iter_start_time = time.time()
        print('Iteration: ' + str(iter))

        # batch_num = int(np.ceil(len(train_data) / float(config.batch_size)))
        random.shuffle(train_data)
        for idx, data in enumerate(train_data):
            if len(data.sentences) == 0:
                continue
            start_time = time.time()
            feature, target, feature_lengths = pair_data_variable(data, vocab_srcs, vocab_para, config)

            model.train()
            optimizer.zero_grad()
            logit = model(feature, feature_lengths)
            loss = F.cross_entropy(logit, target)
            loss_value = loss.data.cpu().numpy()
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), config.clip_norm)
            optimizer.step()

            correct = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
            accuracy = 100.0 * correct

            during_time = float(time.time() - start_time)
            print("Step:{}, Iter:{}, batch:{}, accuracy:{:.4f}, time:{:.2f}, loss:{:.6f}"
                  .format(global_step, iter, idx + 1, accuracy, during_time, loss_value[0]))

            global_step += 1

            if (idx + 1) % config.test_interval == 0 or (idx + 1) == len(train_data):
                if config.para_dev_file:
                    dev_acc, dev_macro = evaluate(model, dev_data, global_step, vocab_srcs, vocab_para, config)
                if config.para_test_file:
                    test_acc, test_macro = evaluate(model, test_data, global_step, vocab_srcs, vocab_para, config)
                if config.para_dev_file:
                    if dev_acc > best_macro:
                        print("Exceed best acc: history = %.2f, current = %.2f" % (best_acc, dev_acc))
                        best_acc = dev_acc
                        if os.path.exists(config.save_model_path):
                            pass
                        else:
                            os.makedirs(config.save_model_path)
                        if -1 < config.save_after <= iter:
                            torch.save(model.state_dict(), os.path.join(config.save_model_path,
                                                                        'model.' + str(global_step)))
                else:
                    if test_macro > best_macro:
                        print("Exceed best acc: history = %.2f, current = %.2f" % (best_macro, test_macro))
                        best_macro = test_macro
                        if os.path.exists(config.save_model_path):
                            pass
                        else:
                            os.makedirs(config.save_model_path)
                        if -1 < config.save_after <= iter:
                            torch.save(model.state_dict(), os.path.join(config.save_model_path,
                                                                        'model.' + str(global_step)))
        during_time = float(time.time() - iter_start_time)
        print('one iter using time: time:{:.2f}'.format(during_time))


def evaluate(model, test_data, step, vocab_srcs, vocab_tgts, config):
    model.eval()
    start_time = time.time()
    corrects, size = 0, 0

    # 初始化宏平均
    macro_averaging = {}
    for i in vocab_tgts.i2w:
        macro_averaging[i] = {'tp': 0, 'fn': 0, 'fp': 0}

    for idx, data in enumerate(test_data):
        if len(data.sentences) == 0:
            continue
        feature, target, feature_lengths = pair_data_variable(data, vocab_srcs, vocab_tgts, config)
        logit = model(feature, feature_lengths)
        correct = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
        corrects += correct
        size += 1

        # 统计宏平均
        gold = target.data
        predict = torch.max(logit, 1)[1].view(target.size()).data
        for idx, i in enumerate(gold):
            if predict[idx] == i:
                macro_averaging[vocab_tgts.id2word(i)]['tp'] += 1
            else:
                macro_averaging[vocab_tgts.id2word(i)]['fn'] += 1
                macro_averaging[vocab_tgts.id2word(predict[idx])]['fp'] += 1
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
    print("Step:{}, accuracy:{:.4f}({}/{}), macro_averaging:{:.4f}, time:{:.2f}"
          .format(step, accuracy, corrects, size, macro_averaging_result, during_time))
    model.train()
    return accuracy, macro_averaging_result


def test(model, data, step, vocab_srcs, vocab_tgts, config):
    model.eval()
    start_time = time.time()
    corrects, size = 0, 0

    for batch in create_batch_iter(data, config.batch_size):
        feature, target, starts, ends, feature_lengths = pair_data_variable(batch,
                                                                            vocab_srcs, vocab_tgts, config)
        logit = model(feature, feature_lengths, starts, ends)
        correct = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
        corrects += correct
        size += len(batch)
    accuracy = 100.0 * corrects / size
    during_time = float(time.time() - start_time)
    print("\nevaluate result: ")
    print("Step:{}, accuracy:{:.4f}({}/{}), time:{:.2f}"
          .format(step, accuracy, corrects, size, during_time))
    model.train()
    return accuracy


def predict(model, data, vocab_srcs, config):
    model.eval()
    feature, starts, ends, feature_lengths = pair_data_variable_predict(data, vocab_srcs, config)
    logit = model(feature, feature_lengths, starts, ends)
    return logit
