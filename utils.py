import torch
import torch.nn as nn
from datetime import timedelta
import time


def sort_by_seq_lens(batch, sequences_lengths, descending=True):
    """
        batch [batch_size, max_seq_len, dim]
        sequences_lengths [batch_size, 1]
    """
    sorted_seq_lens, sorting_index = sequences_lengths.sort(0, descending=descending)
    sorted_batch = batch.index_select(0, sorting_index)
    idx_range = sequences_lengths.new_tensor(torch.arange(0, len(sequences_lengths)))
    _, reverse_mapping = sorting_index.sort(0, descending=False) # 原先所在的位置
    restoration_index = idx_range.index_select(0, reverse_mapping)
    return sorted_batch, sorted_seq_lens, sorting_index, restoration_index

def get_mask(batch, sequences_lengths, max_len = 30):
    # return mask[batch, len] pad - 0 value - 1
    batch_size = batch.size()[0]
    mask = torch.ones(batch_size, max_len, dtype=torch.float)
    mask[batch[:, :max_len] == 0] == 0.0
    return mask

def masked_softmax(tensor, mask):
    """
        tesor [batch_size, q1_max_len, q2_max_len]
        mask [batch_size, len]
    """
    tensor_shape = tensor.size()
    reshaped_tensor = tensor.view(-1, tensor_shape[-1]) #[batch_size * q1_max_len, q2_maxlen]
    while mask.dim() < tensor.dim():
        mask = mask.unsqueeze(1) # [batch_size, 1, q2_max_len]
    mask = mask.expand_as(tensor).contiguous().float() # [batch_size, q1_max_len, q2_max_len]
    reshape_mask = mask.view(-1, mask.size()[-1]) # [batch_size * q1_max_len, q2_max_len]
    result = nn.functional.softmax(reshaped_tensor * reshape_mask, dim=-1) # [batch_size * q1_max_len, q2_max_len]
    result = result * reshape_mask
    result = result / result.sum(dim=-1, keepdim=True) + 1e-13
    return result.view(*tensor_shape)

def weighted_sum(tensor, weights, mask):
    """
        for q2
        tensor [batch_size, q2_len, dim]
        weights [batch_size, q1_len, q2_len]
        mask [batch_size, q1_len]
    """
    weighted_sum = weights.bmm(tensor) # [batch_size, q1_len, dim]
    while mask.dim() < weighted_sum.dim():
        mask = mask.unsqueeze(1)
    mask = mask.transpose(-1, -2) #[batch_size, q1_len, 1]
    mask = mask.expand_as(weighted_sum).contiguous().float() #[batch_size, q1_len, dim]
    return weighted_sum * mask #set pad = 0

def replace_masked(tensor, mask, value):
    mask = mask.unsqueeze(1).transpose(1,2) # [batch_size, max_len, 1]
    reverse_mask =  1.0 - mask
    values_to_add = value * reverse_mask
    return tensor * mask + values_to_add


def get_time_diff(start_time):
    end_time = time.time()
    time_diff = end_time - start_time
    return timedelta(seconds=int(round(time_diff)))


def init_model_weights(module):
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