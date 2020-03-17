from models.biMPM import BIMPM
from models.esim import ESIM
from dataset_iterator import DatasetIterator
import train
from config import Config
from prepocess_data import PrepocessData
import torch

if __name__ == '__main__':
    config = Config()
    prepocess_data = PrepocessData()
    train_data, test_data, word_emb = prepocess_data.build_train_test_data(config)
    word_emb = torch.tensor(word_emb, dtype=torch.float,requires_grad=False).to(config.device)
    config.n_vocab = len(word_emb)
    train_iterator = DatasetIterator(train_data, config.batch_size, config.device)
    test_iterator = DatasetIterator(test_data, config.batch_size, config.device)
    model = ESIM(config.n_vocab, config.hidden_size, config.emb_dim, word_emb, config.n_classes, config.padding_idx, config.dropout, config.lstm_layer, config.device)
    model.to(config.device)
    train.train_model(train_iterator, test_iterator, model, config)