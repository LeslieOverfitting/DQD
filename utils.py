#import tensorflow as tf
import os
import spacy
import numpy as np
#from keras.preprocessing.sequence import pad_sequences
#from collections import Counter
import time
import pickle
import csv
nlp = spacy.load('en_core_web_sm')
UNKNOWN = '<UNK>'
PADDING = '<PAD>'


def feed_data(premise, premise_mask, hypothesis, hypothesis_mask, y_batch,
              dropout_keep_prob):
    feed_dict = {model.premise: premise,
                 model.premise_mask: premise_mask,
                 model.hypothesis: hypothesis,
                 model.hypothesis_mask: hypothesis_mask,
                 model.y: y_batch,
                 model.dropout_keep_prob: dropout_keep_prob}
    return feed_dict


def init_embeddings(vocab, embedding_dims):
    rng = np.random.RandomState(seed = 22)
    random_init_embeddings = rng.normal(size = (len(vocab), embedding_dims))
    return random_init_embeddings.astype(np.float32)

def load_embeddings(path, vocab):
    with open(path, 'rb') as fin:
        _embeddings, _vocab = pickle.load(fin)
    embedding_dims = _embeddings.shape[1]
    embeddings = init_embeddings(vocab, embedding_dims)
    for word, id in vocab.items():
        if word in _vocab:
            embeddings[id] = _embeddings[_vocab[word]]
    return embeddings.astype(np.float32)


def read_data(dataPath):
    with open(dataPath, mode='r', encoding='utf-8') as question_paris:
        reader = csv.DictReader(question_paris)
        data = list(reader)
        data = np.asarray(data)
        np.random.seed(123)
        np.random.shuffle(data)
        length = data.shape[0]
        train = data[:int(0.8 * length)]
        valid = data[int(0.8 * length):int(0.9 * length)]
        test = data[int(0.9 * length):]
        return train, valid, test


def tokenize_data(question_data):
    data = {'q1': [], 'q2': [], 'y':[]}
    for line in question_data:
        data['q1'].append(tokenize_text(line['question1']))
        data['q2'].append(tokenize_text(line['question2']))
        data['y'].append(int(line['is_duplicate']))
    return data


def tokenize_text(text):
    if text is None:
        return []
    text = text.strip().replace('`',"'")
    doc = nlp.tokenizer(text)
    tokens = [token.lower_ for token in doc]
    return tokens

def read_glove(file, dim):
    emb_dict = {}
    dim += 1
    with open(file, encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split(' ')
            if len(tokens) == dim:
                emb_dict[tokens[0]] = list(map(lambda x: float(x), tokens[1:]))
                # {'apple':[]}
    return emb_dict

def token2Index(question, word2Index, glove_emb, question_emb):
    for i in range(len(question)):
        if question[i] not in word2Index:
            if question[i] in glove_emb:
                word2Index[question[i]] = len[word2Index]
                question_emb.append(glove_emb[question[i]])
            else:
                question[i] = UNKNOWN
        question[i] = word2Index[question[i]]


def get_word_embedding(question_data, glove_emd_path, emb_dim):
    print('loading glove embedding....')
    glove_emb = read_glove(glove_emd_path, emb_dim)
    print('loading glove embedding done!')
    word2Index = {PADDING: 0, UNKNOWN: 1}
    question_emb = [np.random.uniform(-0.1, 0.1, emb_dim) for _ in range(2)]
    for data_set in question_data: # train , test, valid
        for question in data_set['q1']:
            token2Index(question, word2Index, glove_emb, question_emb)
        for question in data_set['q2']:
            token2Index(question, word2Index, glove_emb, question_emb)
    print('{} word nums'.format(len(word2Index)))
    question_emb = np.asarray(question_emb, dtype='float32')
    return question_emb, word2Index


def preprocess_data(data_path, glove_path, emb_dim):
    print('loading question data....')
    train, valid, test = read_data(data_path)
    print('loading question data done!')
    print('tokenizing question data....')
    train = tokenize_data(train)
    valid = tokenize_data(valid)
    test = tokenize_data(test)
    question_data = []
    question_data.append(train)
    question_data.append(valid)
    question_data.append(test)
    print('getting question word embedding....')
    question_emd, word2Index = get_word_embedding(question_data, glove_path, emb_dim)
    print('getting question word embedding done!')
    with open('data/data_emb', 'wb') as f:
        pickle.dump((question_data, question_emd, word2Index), f)
    print('Saved.')

def main():
    data_path = os.path.join(os.path.expanduser('~'), 'code/nlp/DQD_model/data/train_dqd.csv')
    glove_path = os.path.join(os.path.expanduser('~'), 'code/embedding/glove.840B.300d.txt')
    preprocess_data(data_path, glove_path, 300)

if __name__ == '__main__':
    main()