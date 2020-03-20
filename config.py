import torch

class Config(object):
    def __init__(self):
        self.data_file_path = 'data/train_dqd.csv'
        self.word_emb_path = 'data/word_emb'
        self.word2Index_path = 'data/word2Index'
        self.model_save_path = 'save_model/esim'
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        #model
        self.n_vocab = 0
        self.batch_size = 64
        self.n_classes = 2
        self.have_len = True
        self.emb_dim = 300
        self.padding_idx = 0
        self.hidden_size = 300
        self.dropout = 0.5
        self.hidden_layer = 2

        # train
        self.learn_rate = 4e-4
        self.epochs_num = 15



