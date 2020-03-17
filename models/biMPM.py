import torch 
import torch.nn as nn
import torch.nn.functional as F

class BIMPM(nn.Module):

    def __init__(self, perspective_len, vocab_num, hidden_size, emb_dim, word_emb, n_classes, padding_idx = 0,dropout = 0.2, lstm_layer = 2, device ='cpu'):
        super(BIMPM, self).__init__()
        self.vocab_num = vocab_num
        self.hidden_size = hidden_size
        self.emb_dim = emb_dim
        self.word_emb = word_emb
        self.n_classes = n_classes
        self.padding_idx = padding_idx
        self.dropout = dropout
        self.lstm_layer = lstm_layer
        self.device = device
        self.perspective_len = perspective_len

        self.encoder = nn.Embedding(self.vocab_num, self.emb_size, padding_idx=self.padding_idx)
        self.encoder.weight.data.copy_(word_emb)
        self.word_emb.weight.requires_grad = False # no fine-tune

 
        # context Representation
        self.context_lstm = nn.LSTM(
            input_size=self.emb_dim,
            hidden_size=self.hidden_size,
            num_layers=self.lstm_layer,
            bidirectional=True,
            batch_first=True,
            bias=True
        )
        
        self.matching_layer = MatchingLayer(self.hidden_size, self.perspective_len)
        # Aggregation Layer
        self.aggregation_lstm = nn.LSTM(
            input_size=self.perspective_len * 8,
            hidden_size=self.perspective_len,
            num_layers=self.lstm_layer,
            bidirectional=True,
            batch_first=True,
            bias=True
        )

        self.pred_fc1 = nn.Linear(self.args.perspective_len * 4, self.args.perspective_len * 2)
        self.pred_fc2 = nn.Linear(self.args.perspective_len * 2, self.args.n_classes)
    
    def forward(self, q1_batch, q2_batch):
        # Word Representation Layer.
        embedded_q1 = self.encoder(q1_batch) # [batch_size, q1_len, emb_dim]
        embedded_q2 = self.encoder(q2_batch)
        # dropout
        embedded_q1 = self.dropout(embedded_q1) # [batch_size, q1_len, emb_dim]
        embedded_q2 = self.dropout(embedded_q2)

        # Context Representation Layer. 
        con_q1, _ = self.context_lstm(embedded_q1, None) # [batch_size, q1_len, hidden_size * 2]
        con_q2, _ = self.context_lstm(embedded_q2, None) # [batch_size, q2_len, hidden_size * 2]
        # dropout
        con_q1 = self.dropout(con_q1)
        con_q2 = self.dropout(con_q2)
        # Matching Layer. 
        m_q1 = self.matching_layer(con_q1, con_q2)
        m_q2 = self.matching_layer(con_q2, con_q1)
        q1_agg, _ = self.aggregation_lstm(m_q1) # [batch, q1_len, hidden_size]
        q2_agg, _ = self.aggregation_lstm(m_q2) # [batch, q2_len, hidden_size]
        fc1_out = self.pred_fc1(torch.cat([q1_agg[:, -1, :self.perspective_len],
                                           q1_agg[:, 0, self.perspective_len:],
                                           q2_agg[:, -1, :self.perspective_len],
                                           q2_agg[:, 0, self.perspective_len:]], dim=-1))
        fc1_out = self.dropout(F.tanh(fc1_out))
        pred = self.pred_fc2(fc1_out)
        return pred


    def dropout(self, v):
        return F.dropout(v, p=self.dropout, training=self.training)



class MatchingLayer(nn.Module):

    def __init__(self, hidden_size, perspective_len = 4):
        super(MatchingLayer, self).__init__()
        self.W = nn.ParameterList([nn.Parameter(torch.randn(perspective_len, hidden_size)) for _ in range(8)])
        self.perspective_len = perspective_len
        self.hidden_size = hidden_size
    

    def forward(self, q1_batch, q2_batch):
        """
            q1_batch [batch, seq_len_q1, hidden_size * 2]
            q2_batch [batch, seq_len_q2, hidden_size * 2]
        """
        batch_size, seq_len_q1, hidden_size = q1_batch.size()
        half = int(batch_size / 2)
        q1_fw, q1_bw = q1_batch.split(half, -1) # [batch, seq_len_q1, hidden_size]
        q2_fw, q2_bw = q2_batch.split(half, -1)

        match_vecs = []
        # Full-Matching
        full_match_fw = full_matching(q1_fw, q2_fw[:, -1,:].repeat(1, seq_len_q1, 1), self.W[0]) # [batch, seq_len_q1, perspective_len]
        full_match_bw = full_matching(q1_bw, q2_bw[:, 0,:].repeat(1, seq_len_q1, 1), self.W[1])
        match_vecs.append(full_match_fw)
        match_vecs.append(full_match_bw)

        # Max-Matching
        max_match_fw = max_pool_matching(q1_fw, q2_fw, self.W[2])
        max_match_bw = max_pool_matching(q1_bw, q2_bw, self.W[3])
        match_vecs.append(max_match_fw)
        match_vecs.append(max_match_bw)
        # Attentive-Matching
        att_fw = attention(q1_fw, q2_fw) # [batch, seq_len_q1, seq_len_q2]
        att_bw = attention(q1_bw, q2_bw)
        # [batch, seq_len_q1, seq_len_q2, 1] * [batch, 1, seq_len_q2, hidden_size] = [batch, seq_len_q1, seq_len_q2, hidden_size]
        attened_q2_fw = att_fw.unsqueeze(3) * q2_fw.unsqueeze(1)
        # [batch, seq_len_q1, seq_len_q2, 1] * [batch, 1, seq_len_q2, hidden_size] = [batch, seq_len_q1, seq_len_q2, hidden_size]
        attened_q2_bw = att_bw.unsqueeze(3) * q2_bw.unsqueeze(1)
        # [batch, seq_len_q2, seq_len_q1, 1] * [batch, 1, seq_len_q1, hidden_size] = [batch, seq_len_q2, seq_len_q1, hidden_size]
        # attened_q1_fw = att_fw.transpose(1, 2).unsqueeze(1) * q1_fw.unsqueeze(1) #
        # attened_q1_bw = att_bw.transpose(1, 2).unsqueeze(1) * q1_bw.unsqueeze(1) #
        # [batch, seq_len_q1,  hidden_size] sum /  [batch, seq_len_q1, 1] = [batch, seq_len_q1, hidden_size]
        attened_mean_q2_fw = div_with_small_value(attened_q2_fw.sum(dim=2), att_fw.sum(dim=2, keepdim=True))
        attened_mean_q2_bw = div_with_small_value(attened_q2_bw.sum(dim=2), att_fw.sum(dim=2, keepdim=True))
        # [batch, seq_len_q2,  hidden_size] sum /  [batch, seq_len_q2, 1] = [batch, seq_len_q2, hidden_size]
        #attened_mean_q1_fw = div_with_small_value(attened_q1_fw.sum(dim=2), att_fw.transpose(1, 2).sum(dim=2, keepdim=True))
        #attened_mean_q1_bw = div_with_small_value(attened_q1_bw.sum(dim=2), att_bw.transpose(1, 2).sum(dim=2, keepdim=True))
        attentive_match_fw = full_matching(q1_fw, attened_mean_q2_fw, self.W[4])
        attentive_match_bw = full_matching(q1_bw, attened_mean_q2_bw, self.W[5])
        match_vecs.append(attentive_match_fw)
        match_vecs.append(attentive_match_bw)
        # max-Attentive-Matching
        # [batch, seq_len_q1, hidden_size]
        attened_q2_fw = attened_q2_fw.max(dim=2)
        attened_q2_bw = attened_q2_bw.max(dim=2) #
        max_attentive_match_fw = full_matching(q1_fw, attened_q2_fw, self.W[6])
        max_attentive_match_bw = full_matching(q1_bw, attened_q2_bw, self.W[7])
        match_vecs.append(max_attentive_match_fw)
        match_vecs.append(max_attentive_match_bw)
        return torch.cat(match_vecs, dim=-1) # [batch, seq_len_q1, 8 * l]





def full_matching(q1, q2, W):
    """
        q1 [batch, seq_len_q1, hidden_size]
        q2 [batch, seq_len_q1, hidden_size]  # q1_len
        w [perspective_len, hidden_size]
    """
    batch, seq_len_q1, hidden_size = q1.size()
    perspective_len = W.size()[0]
    W_rep = W.repeat(batch, seq_len_q1, 1, 1) # [batch, seq_len_q1, perspective_len, hidden_size]
    q1_rep = q1.repeat(perspective_len, 1, 1, 1).permute(1, 2, 0, 3) # [batch, seq_len_q1, perspective_len, hidden_size]
    q2_rep = q2.repeat(perspective_len, 1, 1, 1).permute(1, 2, 0, 3) # [batch, seq_len_q1, perspective_len, hidden_size]
    result = F.cosine_similarity(W_rep * q1_rep, W_rep * q2_rep, dim=-1).view(batch, seq_len_q1, perspective_len) # [batch, seq_len_q1, perspective_len]
    return result

def max_pool_matching(q1, q2, W):
    """
        q1 [batch, seq_len_q1, hidden_size]
        q2 [batch, seq_len_q2, hidden_size]  # q2_len
        w [perspective_len, hidden_size]
    """
    batch, seq_len_q1, hidden_size = q1.size()
    seq_len_q2 = q2.size()[1]
    perspective_len = W.size()[0]
    #W_rep = W.repeat(seq_len_q1, batch, , 1, 1) # [batch, seq_len_q1, perspective_len, hidden_size]
    q1_rep = q1.repeat(perspective_len, 1, 1, 1).permute(2, 1, 0, 3) # [seq_len_q1, batch, perspective_len, hidden_size]
    q2_rep = q2.repeat(perspective_len, 1, 1, 1).permute(2, 1, 0, 3) # [seq_len_q2, batch, perspective_len, hidden_size]
    q1_rep = q1_rep * W # [seq_len_q1, batch, perspective_len, hidden_size]
    q2_rep = q2_rep * W # [seq_len_q2, batch, perspective_len, hidden_size]
    q1_rep = q1.repeat(seq_len_q2, 1, 1, 1, 1).transpose(0, 1) # [seq_len_q1, seq_len_q2, batch, perspective_len, hidden_size]
    consine = F.cosine_similarity(q1_rep, q2_rep, dim=-1).permute(2, 0, 1, 3) # [batch, seq_len_q1, seq_len_q2, perspective_len]
    m_max = consine.max(2) # [batch, seq_len_q1, perspective_len]
    assert m_max.size() == torch.Size([batch, seq_len_q1, perspective_len])
    return m_max

def attention(q1, q2):
    """
        q1 [batch, seq_len_q1, hidden_size]
        q2 [batch, seq_len_q2, hidden_size]  # q2_len
    """
    q1_norm = q1.norm(p=2, dim=2, keepdim=True)#[batch, seq_len_q1, hidden_size]
    q2_norm = q2.norm(p=2, dim=2, keepdim=True).permute(0, 2, 1)#[batch, seq_len_q2, hidden_size]
    molecule = torch.bmm(q1, q2.permute(0, 2, 1))#[batch, seq_len_q1, seq_len_q2]
    return div_with_small_value(molecule, q1_norm * q2_norm)

def div_with_small_value(a, b, eps=1e-8):
    b = b * (b > eps).float() + (b <= eps).float() * eps
    return a / b

def attetive_matching(q1, q2, W):
    """
        q1 [batch, seq_len_q1, hidden_size]
        q2 [batch, seq_len_q2, hidden_size]  # q2_len
        w [perspective_len, hidden_size]
    """
    batch, seq_len_q1, hidden_size = q1.size()
    seq_len_q2 = q2.size()[1]
    perspective_len = W.size()[0]