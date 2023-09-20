from crf import CRF
from torch import nn
import torch.nn.functional as F

from allennlp.modules.elmo import Elmo

import ipdb

class Lstm(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(p=config.dropout)
        self.word_lstm = nn.LSTM(config.dim_elmo, config.hidden_size_lstm, bidirectional=True)

    def forward(self, word_input):
        # Word_dim = (batch_size x seq_length)
        word_emb = self.dropout(word_input)
        output, (h, c) = self.word_lstm(word_emb) #shape = S*B*hidden_size_lstm
        return output, (h,c)

class SequenceTagger(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.lstm = Lstm(config)
        self.crf = CRF(config.ntags)
        self.linear = LinearClassifier(config, layers=[config.hidden_size_lstm*2, config.ntags], drops=[0.5])
        self.dropout = nn.Dropout(p=config.dropout)

    def forward(self, word_input, mask=None, labels=None):
        # Word_dim = (batch_size x seq_length)
        if mask is None:
            print("mAAAAAsk!!!")
            return -1

        # pdb.set_trace()
        lstm_output, (_, _) = self.lstm(word_input) #shape = S*B*hidden_size_lstm
        lstm_output = self.dropout(lstm_output)
        lstm_output = self.linear(lstm_output)
        preds = self.crf.decode(lstm_output.transpose(0,1), mask=mask.transpose(0,1))
        if labels is not None:
            loss = -1 * self.crf(lstm_output.transpose(0,1), labels.transpose(0,1), mask=mask.transpose(0,1))
            return loss, preds

        return preds #shape = S*B*ntags

class LinearBlock(nn.Module):
    def __init__(self, ni, nf, drop):
        super().__init__()
        self.lin = nn.Linear(ni, nf)
        self.drop = nn.Dropout(drop)
        self.bn = nn.BatchNorm1d(ni)

    def forward(self, x):
        return self.lin(self.drop(self.bn(x)))


class LinearClassifier(nn.Module):
    def __init__(self, config, layers, drops):
        self.config = config
        super().__init__()
        self.layers = nn.ModuleList([
            LinearBlock(layers[i], layers[i + 1], drops[i]) for i in range(len(layers) - 1)])

    def forward(self, input):
        output = input
        sl,bs,_ = output.size()
        x = output.view(-1, 2*self.config.hidden_size_lstm)

        for l in self.layers:
            l_x = l(x)
            x = F.relu(l_x)
        return l_x.view(sl, bs, self.config.ntags)
