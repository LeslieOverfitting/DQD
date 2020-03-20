import spacy
import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm

nlp = spacy.load('en_core_web_sm')

class PrepocessData:
    def __init__(self):
        pass

    def read_glove(self, file_path, dim=300):
        print('read glove')
        emb_dict = {}
        with open(file_path, encoding='utf-8') as f:
            for line in tqdm(f):
                tokens = line.strip().split(' ')
                emb_dict[tokens[0]] = list(map(lambda x: float(x), tokens[1:]))
        print('read glove over')
        return emb_dict

    def read_data(self, data_path, max_len =30):
        data_df = pd.read_csv(data_path)
        data_df = data_df[['id', 'question1', 'question2', 'is_duplicate']]
        data_df['question1'].apply(str)
        data_df['question1_len'] = data_df['question1'].apply(lambda x: min(max_len, len(x)))
        data_df['question2'].apply(str)
        data_df['question2_len'] = data_df['question2'].apply(lambda x: min(max_len, len(x)))
        data_df['id'].apply(str)
        data_df['is_duplicate'].apply(int)
        data = {
            "id": data_df['id'].values.tolist(),
            "question1": data_df['question1'].values.tolist(),
            "question2": data_df['question2'].values.tolist(),
            "question1_len": np.asarray(data_df['question1_len'].values.tolist(), dtype=np.int32),
            "question2_len": np.asarray(data_df['question2_len'].values.tolist(), dtype=np.int32),
            "label": np.asarray(data_df['is_duplicate'].values.tolist(), dtype=np.float32),
        }
        data['question1'] = list(map(self.tokenize_sentence, data['question1']))
        data['question2'] = list(map(self.tokenize_sentence, data['question2']))
        return data    
    def tokenize_sentence(self, sentence):
        if sentence is None:
            return []
        sentence = sentence.strip().replace('`',"'")
        doc = nlp.tokenizer(sentence)
        tokens = [token.lower_ for token in doc]
        return tokens

    def build_word_embedding_bysentetnce(self, sentence, pre_train_emb, word2Index, word_emb):
        for word in sentence:
            if word not in word2Index:
                if word in pre_train_emb:
                    word2Index[word] = len(word2Index)
                    word_emb.append(pre_train_emb[word])
                else:
                    pass
            else:
                pass    

    def build_word_embedding(self, data, pre_train_path, dim = 300):
        pre_train_emb = self.read_glove(pre_train_path)
        word2Index = {'PAD': 0, 'UNK':1}
        word_emb = [np.random.uniform(-0.1, 0.1, dim) for _ in range(2)]
        question1 = data["question1"]
        question2 = data["question2"]
        for i in tqdm(range(len(question1))):
            self.build_word_embedding_bysentetnce(question1[i], pre_train_emb, word2Index, word_emb)
            self.build_word_embedding_bysentetnce(question2[i], pre_train_emb, word2Index, word_emb)
        word_emb = np.asarray(word_emb, dtype='float32')
        return word_emb, word2Index

    def build_train_test_data(self, config):
        data = self.read_data(config.data_file_path)
        padded_data = self.padding_question(data, 30)
        word2Index = self.load_word2Index(config.word2Index_path)
        word_emb = self.load_word_emb(config.word_emb_path)
        self.transform_to_indices(data, word2Index)
        data_indexed = list(zip(data['question1'], data['question1_len'], data['question2'], data['question2_len'], data['label']))
        train_data, test_data = self.split_dataset(data_indexed)
        return train_data, test_data, word_emb

    def split_dataset(self, data):
        np.random.seed(123)
        np.random.shuffle(data)
        length = len(data)
        train_set = data[:int(length * 0.9)]
        test_set = data[int(length * 0.9): ]
        return train_set, test_set

    def transform_to_indices(self, data, word2Index):
        data['question1'] = list(map(lambda question: [word2Index.get(word, 1) for word in question], data['question1']))
        data['question2'] = list(map(lambda question: [word2Index.get(word, 1) for word in question], data['question2']))

    def padding_question(self, data, max_len=30):
        data['question1'] = list(map(lambda question: question[:max_len] + ['PAD'] * max(0, max_len - len(question)), data['question1']))
        data['question2'] = list(map(lambda question: question[:max_len] + ['PAD'] * max(0, max_len - len(question)), data['question2']))

    def save_word_emb(self, word_emb, word_emb_path):
        with open(word_emb_path, mode='wb') as f:
            pickle.dump(word_emb, f)
        print('word emb saved')

    def load_word_emb(self, word_emb_path):
        with open(word_emb_path, mode='rb') as f:
            word_emb = pickle.load(f)
        return word_emb
    
    def save_word2Index(self, word2Index, word2Index_path):
        with open(word2Index_path, mode='wb') as f:
            pickle.dump(word2Index, f)
        print('word2Index saved')

    def load_word2Index(self, word2Index_path):
        with open(word2Index_path, mode='rb') as f:
            word2Index = pickle.load(f)
        return word2Index
   
