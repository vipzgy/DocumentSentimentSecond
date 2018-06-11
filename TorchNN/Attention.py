# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable


class Attention(nn.Module):
    def __init__(self, origin_size, attention_size, use_cuda):
        super(Attention, self).__init__()
        self.origin_size = origin_size
        self.attention_size = attention_size
        self.use_cuda = use_cuda

        self.w = nn.Linear(origin_size, attention_size, bias=True)
        init.xavier_uniform(self.w.weight)
        self.u = nn.Linear(attention_size, 1, bias=False)
        init.xavier_uniform(self.u.weight)

    def forward(self, h, h_lengths):
        h = torch.transpose(h, 0, 1)

        s_max = max([l for l in h_lengths])
        s_mask = []
        for idx in range(len(h_lengths)):
            s_t_mask = []
            for idy in range(s_max):
                if idy < h_lengths[idx]:
                    s_t_mask.append(1)
                else:
                    s_t_mask.append(0)
            s_mask.append(s_t_mask)

        if self.use_cuda:
            h_mask = Variable(torch.IntTensor(s_mask)).cuda()
            zeros_hidden = Variable(torch.zeros(h.size(0), h.size(1), self.origin_size)).cuda()
            zeros_attention = Variable(torch.zeros(h.size(0), h.size(1), self.attention_size)).cuda()
            h_mask = torch.abs(h_mask - 1).type(torch.cuda.ByteTensor)
        else:
            h_mask = Variable(torch.IntTensor(s_mask))
            zeros_hidden = Variable(torch.zeros(h.size(0), h.size(1), self.origin_size))
            zeros_attention = Variable(torch.zeros(h.size(0), h.size(1), self.attention_size))
            h_mask = torch.abs(h_mask - 1).type(torch.ByteTensor)
        h_mask = torch.unsqueeze(h_mask, 2)

        h = h.masked_scatter(h_mask, zeros_hidden.masked_select(h_mask))

        h = self.w(h)
        h = F.tanh(h)
        h = h.masked_scatter(h_mask, zeros_attention.masked_select(h_mask))
        beta = self.u(h)

        beta = torch.squeeze(beta, 2)
        if self.use_cuda:
            beta_0 = Variable(torch.zeros(beta.size(0), beta.size(1))).cuda()
        else:
            beta_0 = Variable(torch.zeros(beta.size(0), beta.size(1)))
        beta_0 += -1e20
        h_mask = torch.squeeze(h_mask, 2)
        if self.use_cuda:
            beta = beta.masked_scatter(h_mask, beta_0.masked_select(h_mask))
        else:
            beta = beta.masked_scatter(h_mask, beta_0.masked_select(h_mask))
        alpha = F.softmax(beta, dim=1)

        # 这里就可以作图了，看看attention的值怎么样

        alpha = torch.unsqueeze(alpha, 2)
        alpha = alpha.repeat(1, 1, self.attention_size)
        s = torch.mul(h, alpha)
        s = torch.sum(s, 1)
        return s
