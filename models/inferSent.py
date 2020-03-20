import torch
import torch.nn as nn
import torch.nn.functional as F



class InferSent(nn.Module):

    def __init__(self, config, word_emb):
        super(InferSent, self).__init__()
        self.n_vocab = config.n_vocab
        self.emb_dim = config.emb_dim
        self.word_emb = word_emb
        self.hidden_size = config.hidden_size
        self.hidden_layer = config.hidden_layer
        self.n_classes = config.n_classes
        self.dropout = config.dropout
        self.padding_idx = config.padding_idx
        self.device = config.device
        self.word_encoder = nn.Embedding(self.n_vocab, self.emb_dim, padding_idx=self.padding_idx)
        self.word_encoder.weight.data.copy_(self.word_emb)
        self.word_encoder.weight.requires_grad = False # no fine-tune
        self.lstm_encoder = nn.LSTM(
            input_size=self.emb_dim,
            hidden_size=self.hidden_size,
            num_layers=self.hidden_layer,
            bias=True,
            bidirectional=True
        )
        self.mlp = nn.Sequential(nn.Dropout(p=self.dropout),
                                nn.Linear(4 * 2 * self.hidden_size, self.hidden_size),
                                nn.Tanh(),
                                nn.Dropout(p=self.dropout),
                                nn.Linear(self.hidden_size, self.n_classes)
                                )
        self.apply(_init_inferSent_weights)


    def forward(self, inputs):
        """
            input:
                question1: [batch_size, seq1_len]
                question1_len: [batch_size]
                question2: [batch_size, seq2_len]
                question2_len: [batch_size]
        """
        question1, question1_len, question2, question2_len = inputs
        q1_mask = get_mask(question1).to(self.device) # [batch_size, seq1_len]
        q2_mask = get_mask(question2).to(self.device) # [batch_size, seq2_len]

        embeded_q1 = self.word_encoder(question1)# [batch_size, seq1_len, emb_dim]
        embeded_q2 = self.word_encoder(question2)# [batch_size, seq2_len, emb_dim]

        # encoder lstm
        sorted_q1_batch, sorted_q1_lens, _, restoration_q1 = sort_by_seq_lens(embeded_q1, question1_len)
        sorted_q2_batch, sorted_q2_lens, _, restoration_q2 = sort_by_seq_lens(embeded_q2, question2_len)
        # pack sequence
        packed_q1_batch = nn.utils.rnn.pack_padded_sequence(sorted_q1_batch, sorted_q1_lens, batch_first=True)
        packed_q2_batch = nn.utils.rnn.pack_padded_sequence(sorted_q2_batch, sorted_q2_lens, batch_first=True)
        # lstm
        lstm_out_q1, _ = self.lstm_encoder(packed_q1_batch, None)
        lstm_out_q2, _ = self.lstm_encoder(packed_q2_batch, None)
        # pad sequence
        pad_q1, _ = nn.utils.rnn.pad_packed_sequence(lstm_out_q1, batch_first=True)
        pad_q2, _ = nn.utils.rnn.pad_packed_sequence(lstm_out_q2, batch_first=True)
        # reorder
        encoded_q1 = pad_q1.index_select(0, restoration_q1)# [batch_size, seq1_len, hidden_size * 2]
        encoded_q2 = pad_q2.index_select(0, restoration_q2)# [batch_size, seq2_len, hidden_size * 2]
        
        # max pool
        encoded_q1[encoded_q1 == 0] = -1e9 # [batch_size, seq1_len, hidden_size * 2]
        encoded_q2[encoded_q2 == 0] = -1e9
        u = torch.max(encoded_q1, 1)[0]
        v = torch.max(encoded_q2, 1)[0]
        # (u, v, |u − v|, u ∗ v)
        merge = torch.cat([u, v, torch.abs(u - v), u * v], dim=1)
        logits = self.mlp(merge)
        return logits



def get_mask(batch_sequence):
    """
        input:
            batch_sequence: [batch_size, seq1_len]
    """
    batch_size, padded_len = batch_sequence.size()
    mask = torch.ones(batch_size, padded_len, dtype=torch.float)
    mask[batch_sequence[:, :padded_len] == 0] = 0.0
    return mask

def sort_by_seq_lens(batch_sequence, batch_length):
    """
        input:
            batch_sequence: [batch_size, seq1_len, emb_dim]
            batch_length: [batch_size, 1]
    """
    sorted_seq_lens, sorted_seq_index = batch_length.sort(0, descending=True)
    sorted_batch = batch_sequence.index_select(0, sorted_seq_index)
    _, reverse_mapping = sorted_seq_index.sort(0, descending=False)
    idx_range = batch_length.new_tensor(torch.arrange(0, len(batch_length)))
    restoration_index = idx_range.index_select(0, reverse_mapping)
    return sorted_batch, sorted_seq_lens, sorted_seq_index, restoration_index



def _init_inferSent_weights(module):
    """
    Initialise the weights of the inferSent model.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight.data)
        nn.init.constant_(module.bias.data, 0.0)

    elif isinstance(module, nn.LSTM):
        for name, param in module.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
