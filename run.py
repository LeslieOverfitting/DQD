from models.biMPM import BIMPM
from models.esim import ESIM
from models.sse import SSE
from models.inferSent import InferSent
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
    #model = ESIM(config, word_emb)
    #model = InferSent(config, word_emb)
    #model = SSE(config, word_emb)
    model = BIMPM(config, word_emb, 4)
    model.to(config.device)
    print(model)
    train.train_model(train_iterator, test_iterator, model, config)