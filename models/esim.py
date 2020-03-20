import torch 
import torch.nn as nn
import torch.nn.functional as F
from utils import sort_by_seq_lens, get_mask, masked_softmax, weighted_sum, replace_masked, init_model_weights
from models.seq2SeqEncoder import Seq2SeqEncoder

# widely inspired from https://github.com/coetaur0/ESIM/tree/e905a2f2891c64613b2d6d46635504bd2827f1e3

class ESIM(nn.Module):

    def __init__(self, config, word_emb):
        super(ESIM, self).__init__()
        self.n_vocab = config.n_vocab
        self.hidden_size = config.hidden_size
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
        self._encodding = Seq2SeqEncoder(self.emb_dim, self.hidden_size,  num_layers=self.hidden_layer, bias=True, dropout=self.dropout)

        self._projection = nn.Sequential(nn.Linear(4 * 2 * self.hidden_size, self.hidden_size),
                                        nn.ReLU())

        self._composition_lstm = Seq2SeqEncoder(self.hidden_size, self.hidden_size, num_layers=self.hidden_layer, bias=True, dropout=self.dropout)
        self._predict_fc = nn.Sequential(nn.Dropout(p=self.dropout),
                                            nn.Linear(4 * 2 * self.hidden_size, self.hidden_size),
                                            nn.Tanh(),
                                            nn.Dropout(p=self.dropout),
                                            nn.Linear(self.hidden_size, self.n_classes)
                                            )
        self.apply(init_model_weights)

    def forward(self,inputs):
        """
            question1s [batch, max_len, dim]
            question1_lengths [batch, len] 
            question2s [batch, max_len, dim]
            question2_lengths [batch, len]
        """
        question1s, question1_lengths, question2s, question2_lengths = inputs
        # input encoding
        q1_mask = get_mask(question1s, question1_lengths).to(self.device) # [batch_size, len] 
        q2_mask = get_mask(question2s, question2_lengths).to(self.device)
        
        embedded_q1 = self.encoder(question1s)
        embedded_q2 = self.encoder(question2s)
        encoded_q1 = self._encodding(embedded_q1, question1_lengths) # [batch_size, max_len_q1, dim]
        encoded_q2 = self._encodding(embedded_q2, question2_lengths) # [batch_size, max_len_q2, dim]
        # local inference
        # e_ij = a_i^Tb_j  (11)
        similarity_matrix = encoded_q1.bmm(encoded_q2.transpose(2, 1).contiguous()) # [batch_size, max_len_q1, max_len_q2]
        q1_q2_atten = masked_softmax(similarity_matrix, q2_mask)  # [batch_size, max_len_q1, max_len_q2]
        q2_q1_atten = masked_softmax(similarity_matrix.transpose(2, 1).contiguous(), q1_mask)
        
        # eij * bj
        a_hat = weighted_sum(encoded_q1, q1_q2_atten, q1_mask) # [batch_size, max_len_q1, dim]
        b_hat = weighted_sum(encoded_q2, q2_q1_atten, q2_mask) # [batch_size, max_len_q2, dim]

        # Enhancement of local inference information
        # ma = [a¯; a~; a¯ − a~; a¯ a~];
        # mb = [b¯; b~; b¯ − b~; b¯ b~]
        m_a = torch.cat([encoded_q1, a_hat, encoded_q1 - a_hat, encoded_q1 * a_hat], dim=-1) # [batch_size, max_len_q1, 4 * dim]
        m_b = torch.cat([encoded_q2, b_hat, encoded_q2 - b_hat, encoded_q2 * b_hat], dim=-1)

        # 3.3 Inference Composition
        projected_q1 = self._projection(m_a)  # [batch_size, max_len_q1, dim]
        projected_q2 = self._projection(m_b)  # [batch_size, max_len_q2, dim]
        v_a = self._composition_lstm(projected_q1, question1_lengths) # [batch_size, max_len_q1, dim]
        v_b = self._composition_lstm(projected_q2, question2_lengths) # [batch_size, max_len_q2, dim]
        v_a_avg = torch.sum(v_a * q1_mask.unsqueeze(1).transpose(2, 1), dim=1)  \
                   / torch.sum(q1_mask, dim=1, keepdim = True) # q1_mask batch_size, 1, max_len_q1
        v_b_avg = torch.sum(v_b * q2_mask.unsqueeze(1).transpose(2, 1), dim=1) \
                   / torch.sum(q2_mask, dim=1, keepdim = True)
        v_a_max, _ = replace_masked(v_a, q1_mask, -1e7).max(dim=1) # [batch_size, dim]
        v_b_max, _ = replace_masked(v_b, q2_mask, -1e7).max(dim=1)

        v = torch.cat([v_a_avg, v_a_max, v_b_avg, v_b_max], dim=1) # [batch_size, dim * 4]

        logits = self._predict_fc(v)
        return logits


