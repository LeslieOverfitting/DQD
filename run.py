import numpy as np
from importlib import import_module
from models.esim import ESIM
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from datetime import datetime
import argparse
from utils import *
import time
import os

UNKNOWN = '<UNK>'
PADDING = '<PAD>'

# parser = argparse.ArgumentParser(description="duplicate question detection")
# parser.add_argument('--model', type=str, required=True, help='choose a model: ')

def feed_data(q1_pad, q1_mask, q2_pad, q2_mask, y_batch,
              dropout_keep_prob = 0.5):
    feed_dict = {model.premise: q1_pad,
                 model.premise_mask: q1_mask,
                 model.hypothesis: q2_pad,
                 model.hypothesis_mask: q2_mask,
                 model.y: y_batch,
                 model.dropout_keep_prob: dropout_keep_prob
                 }
    return feed_dict

def prepare_input_data(question_data, word2Index, max_len = 100):
    q1_list, q2_list = [], []
    q1_mask, q2_mask = [], []
    length = len(question_data['q1'])
    for i in range(length):
        q1_temp = question_data['q1'][i]
        q2_temp = question_data['q2'][i]
        if len(q1_temp) > max_len:
            q1_temp = q1_temp[:max_len]
        if len(q2_temp) > max_len:
            q2_temp = q2_temp[:max_len]
        
        q1_mask.append(len(q1_temp))    
        q2_mask.append(len(q2_temp))
        q1_list.append([word2Index[word] if word in word2Index else word2Index[UNKNOWN] for word in q1_temp])
        q2_list.append([word2Index[word] if word in word2Index else word2Index[UNKNOWN] for word in q2_temp])

    q1_pad = pad_sequences(q1_list, max_len, padding='post')
    q2_pad = pad_sequences(q2_list, max_len, padding='post')
    y = np.asarray(question_data['y'], np.int32)
    q1_mask = np.asarray(q1_mask, np.int32)
    q2_mask = np.asarray(q2_mask, np.int32)
    return q1_pad, q1_mask, q2_pad, q2_mask, y

def next_batch(q1_list, q1_mask, q2_list, q2_mask, y, batch_size = 8, shuffle = True):
    sample_num = len(q1_list)
    batch_num = int((sample_num - 1) / batch_size) + 1
    if shuffle:
        indices = np.random.permutation(np.arange(sample_num))
        q1_list = q1_list[indices]
        q1_mask = q1_mask[indices]
        q2_list = q2_list[indices]
        q2_mask = q2_mask[indices]
        y = y[indices]
    
    for i in range(batch_num):
        start_index = i * batch_size
        end_index = min(start_index + batch_size, sample_num)
        yield(q1_list[start_index: end_index], 
              q1_mask[start_index: end_index],
              q2_list[start_index: end_index], 
              q2_mask[start_index: end_index],
              y[start_index: end_index]
            )

def evaluate(sess, q1_pad, q1_mask, q2_pad, q2_mask, y):
    batches = next_batch(q1_pad, q1_mask, q2_pad, q2_mask, y)
    data_nums = len(q1_pad)
    total_loss = 0.0
    total_acc = 0.0
    for batch in batches:
        batch_nums = len(batch[0])
        feed_dict = feed_data(*batch, 1.0)
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_nums
        total_acc += acc * batch_nums
    return total_loss / data_nums, total_acc / data_nums



def train(question_data, question_emd, word2Index):
    print("begin train model")
    start_time = time.time()
    q1_pad_train, q1_mask_train, q2_pad_train, q2_mask_train, y_train = prepare_input_data(question_data[0], word2Index)
    q1_pad_val, q1_mask_val, q2_pad_val, q2_mask_val, y_val = prepare_input_data(question_data[1], word2Index)
    q1_pad_test, q1_mask_test, q2_pad_test, q2_mask_test, y_test = prepare_input_data(question_data[2], word2Index)
    train_data_num = len(q1_pad_train)
    time_diff = get_time_diff(start_time)
    print("prepare input data done", time_diff)
    saver = tf.train.Saver(max_to_keep=5)
    save_file_dir = 'saveModel'
    if not os.path.exists(save_file_dir):
        os.makedirs(save_file_dir)
    # init graph 
    print("begin training")
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    start_time = time.time()
    for epoch in range(100):
        print('Epoch : ', epoch + 1) 
        batches = next_batch(q1_pad_train, q1_mask_train, q2_pad_train, q2_mask_train, y_train)
        total_loss, total_acc = 0.0, 0.0
        for batch in batches:
            batch_nums = len(batch[0])
            feed_dict = feed_data(*batch)
            _, batch_loss, batch_acc = sess.run([model.train, model.loss, model.acc], feed_dict=feed_dict)
            total_loss += batch_loss * batch_nums
            total_acc += batch_acc * batch_nums
        
        # evaluate on valid date set
        if((epoch + 1) % 5 == 0):
            loss_val, acc_val = evaluate(sess, q1_pad_val, q1_mask_val, q2_pad_val, q2_mask_val, y_val)
            saver.save(sess = sess, save_path = 'saveModel/' + 'esim_dev_loss_{:.4f}.ckpt'.format(loss_val))
            time_diff = get_time_diff(start_time)
            msg = 'Epoch : {0:>3}, Train Batch Loss : {2:>6.2}, Train Batch Acc : {3:>6.2%}, Dev Loss : {4:>6.2}, Dev Acc : {5:>6.2%}'
            print(msg.format(epoch + 1, total_loss / train_data_num, total_acc / train_data_num, loss_val, acc_val))
    print("train over")

        
if __name__ == '__main__':
    seq_length = 100
    embedding_size = 300
    hidden_size = 300
    attention_size = 300
    batch_size = 8
    learning_rate = 0.001
    optimizer = 'adam'
    l2 = 0.0
    clip_value = 10
    question_data, question_emd, word2Index = load_prepocess_data('data/question_data_emb')
    print("question_emd shape:", question_emd.shape[0])
    n_vocab = question_emd.shape[0]
    tf.reset_default_graph()
    model = ESIM(seq_length, n_vocab, embedding_size, hidden_size, attention_size, 1,\
                 batch_size, learning_rate, optimizer, l2, clip_value, question_emd)
    
    train(question_data, question_emd, word2Index)