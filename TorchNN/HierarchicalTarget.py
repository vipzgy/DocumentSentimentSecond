# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from .Attention import Attention
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class HierarchicalTarget(nn.Module):
    def __init__(self, config, embed_size, embed_dim, padding_idx, label_size, embedding=None):
        super(HierarchicalTarget, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(embed_size, embed_dim, padding_idx=padding_idx)
        if embedding is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embedding))
        self.dropout = nn.Dropout(config.dropout_embed)

        self.s_lstm = nn.LSTM(embed_dim, config.s_hidden_size, num_layers=config.s_num_layers,
                              dropout=config.s_dropout_rnn, batch_first=True, bidirectional=True)
        self.s_attention = Attention(config.s_hidden_size * 2, config.s_attention_size, config.use_cuda)

        self.p_lstm = nn.LSTM(config.s_attention_size, config.p_hidden_size, num_layers=config.p_num_layers,
                              dropout=config.p_dropout_rnn, bidirectional=True)
        self.p_attention = Attention(config.p_hidden_size * 2, config.p_attention_size, config.use_cuda)

        self.output_linear = nn.Linear(config.p_attention_size, label_size)

    def forward(self, inputs_words, inputs_labels, inputs_lengths):
        all_x = []

        for idx, (x, x_labels) in enumerate(zip(inputs_words, inputs_labels)):
            x = self.embedding(x)
            x = self.dropout(x)
            x_labels = self.embedding(x_labels)
            x = x + x_labels

            x = pack_padded_sequence(x, inputs_lengths[idx])
            x, _ = self.s_lstm(x)
            x, _ = pad_packed_sequence(x)
            all_x.append(x)

        # 这个时候batch_size要不要增大到句子个数
        # 先试一试batch_size大于句子最大长度，设为150吧
        x = all_x[0]
        s = self.s_attention(x, inputs_lengths[0])
        s = s.unsqueeze(0)
        s, _ = self.p_lstm(s)
        s = torch.transpose(s, 0, 1)
        p = self.p_attention(s, [s.size(0)])

        p = self.output_linear(p)
        return p
