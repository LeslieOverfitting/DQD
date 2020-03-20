import torch 
import torch.nn as nn
import torch.nn.functional as F
from utils import sort_by_seq_lens, get_mask, init_model_weights
from models.seq2SeqEncoder import Seq2SeqEncoder

class SSE(nn.Module):

    def __init__(self, config, word_emb, hidden_size=[256, 512, 1024]):
        super(SSE, self).__init__()
        self.n_vocab = config.n_vocab
        self.hidden_size = hidden_size
        self.emb_dim = config.emb_dim
        self.word_emb = word_emb
        self.n_classes = config.n_classes
        self.padding_idx = config.padding_idx
        self.dropout = config.dropout
        self.hidden_layer = config.hidden_layer
        self.device = config.device
        self.encoder = nn.Embedding(self.n_vocab, self.emb_dim, padding_idx=self.padding_idx)
        if self.word_emb is not None:
            self.encoder.weight.data.copy_(self.word_emb)
        
        self._shortcut_lstm1 = Seq2SeqEncoder(self.emb_dim, self.hidden_size[0],  num_layers=1, bias=True, dropout=self.dropout)
        self._shortcut_lstm2 = Seq2SeqEncoder(self.emb_dim + 2 * self.hidden_size[0], self.hidden_size[1],  num_layers=1, bias=True, dropout=self.dropout)
        self._shortcut_lstm3 = Seq2SeqEncoder(self.emb_dim + 2 * self.hidden_size[0] + 2 * self.hidden_size[1], self.hidden_size[2],  num_layers=1, bias=True, dropout=self.dropout)
        self.mlp = nn.Sequential(nn.Dropout(p=self.dropout),
                                nn.Linear(self.hidden_size[2] * 2 * 4, self.n_classes))

        self.apply(init_model_weights)

    def forward(self, inputs):
        """
            question1s [batch, max_len]
            question1_lengths [batch, len] 
            question2s [batch, max_len]
            question2_lengths [batch, len]
        """
        question1s, question1_lengths, question2s, question2_lengths = inputs
        # input encoding
        embedded_q1 = self.encoder(question1s)# [batch_size, max_len_q1,emb_dim]
        embedded_q2 = self.encoder(question2s)# [batch_size, max_len_q2, emb_dim]
        # shortcut lstm1
        q1_lstm1_outputs = self._shortcut_lstm1(embedded_q1, question1_lengths) # [batch_size, max_len_q1, hidden_size[0]]
        q2_lstm1_outputs = self._shortcut_lstm1(embedded_q2, question2_lengths) # [batch_size, max_len_q2, hidden_size[0]]
        q1_lstm2_inputs = torch.cat([embedded_q1, q1_lstm1_outputs], dim=-1)# [batch_size, max_len_q1, emb_dim + 2*hidden_size[0]]
        q2_lstm2_inputs = torch.cat([embedded_q2, q2_lstm1_outputs], dim=-1)
        
        # shortcut lstm2
        q1_lstm2_outputs = self._shortcut_lstm2(q1_lstm2_inputs, question1_lengths) # [batch_size, max_len_q1, 2*hidden_size[1]]
        q2_lstm2_outputs = self._shortcut_lstm2(q2_lstm2_inputs, question2_lengths) # [batch_size, max_len_q2, 2*hidden_size[1]]
        q1_lstm3_inputs = torch.cat([embedded_q1, q1_lstm1_outputs, q1_lstm2_outputs], dim=-1)# [batch_size, max_len_q1, emb_dim + 2*hidden_size[0] + 2*hidden_size[1]]
        q2_lstm3_inputs = torch.cat([embedded_q2, q2_lstm1_outputs, q2_lstm2_outputs], dim=-1)

        # shortcut lstm3
        encoded_q1 = self._shortcut_lstm3(q1_lstm3_inputs, question1_lengths) # [batch_size, max_len_q1, hidden_size[2]*2]
        encoded_q2 = self._shortcut_lstm3(q2_lstm3_inputs, question2_lengths) # [batch_size, max_len_q2, hidden_size[2]*2]

        # max pool 
        encoded_q1[encoded_q1 == 0] == -1e9
        encoded_q2[encoded_q2 == 0] == -1e9
        max_encoded_q1 = torch.max(encoded_q1, dim=1)[0] # [batch_size, hidden_size[2] * 2]
        max_encoded_q2 = torch.max(encoded_q2, dim=1)[0]

        # combine m = [vp; vh; jvp − vhj ; vp ⊗ vh]

        m = torch.cat([max_encoded_q1, max_encoded_q2, torch.abs(max_encoded_q1 - max_encoded_q2), max_encoded_q1 * max_encoded_q2], dim=1)

        predict = self.mlp(m)
        return predict

