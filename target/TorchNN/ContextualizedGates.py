# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from .BILSTM import BILSTM
from .Attention import Attention


class ContextualizedGates(nn.Module):
    def __init__(self, config, embed_size, embed_dim, padding_idx, label_size, embedding=None):
        super(ContextualizedGates, self).__init__()
        self.config = config
        self.bilstm = BILSTM(config, embed_size, embed_dim, padding_idx, embedding)
        self.attention_s = Attention(config)
        self.attention_left = Attention(config)
        self.attention_right = Attention(config)

        self.w1 = nn.Linear(config.hidden_size * 4, config.hidden_size * 2)
        init.xavier_uniform(self.w1.weight)
        self.w2 = nn.Linear(config.hidden_size * 4, config.hidden_size * 2)
        init.xavier_uniform(self.w2.weight)
        self.w3 = nn.Linear(config.hidden_size * 4, config.hidden_size * 2)
        init.xavier_uniform(self.w3.weight)

        self.w4 = nn.Linear(config.hidden_size * 2, label_size)
        init.xavier_uniform(self.w4.weight)

    def forward(self, w, length, start, end):
        s_slice, targeted_slice, left_slice, right_slice, \
            s_mask, targeted_mask, left_mask, right_mask = self.bilstm(w, length, start, end)

        if s_slice is not None:
            s = self.attention_s(s_slice, s_mask, targeted_slice)
        if left_slice is not None:
            sl = self.attention_left(left_slice, left_mask, targeted_slice)
        if right_slice is not None:
            sr = self.attention_right(right_slice, right_mask, targeted_slice)

        m_list = []
        ht = torch.mean(targeted_slice, 1)
        if s_slice is not None:
            z = self.w1(torch.cat([s, ht], 1))
            m_list.append(z)
        if left_slice is not None:
            zl = self.w2(torch.cat([sl, ht], 1))
            m_list.append(zl)
        if right_slice is not None:
            zr = self.w3(torch.cat([sr, ht], 1))
            m_list.append(zr)

        count = 0
        zz = torch.cat(m_list, 0)
        zz = F.softmax(zz, 0)
        if s_slice is not None:
            z = torch.squeeze(zz[count], 0)
            ss = torch.mul(z, s)
            count += 1
        if left_slice is not None:
            zl = torch.squeeze(zz[count], 0)
            ss += torch.mul(zl, sl)
            count += 1
        if right_slice is not None:
            zr = torch.squeeze(zz[count], 0)
            ss += torch.mul(zr, sr)

        result = self.w4(ss)
        return result
