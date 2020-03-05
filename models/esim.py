import tensorflow as tf 
from tensorflow.contrib.rnn import LSTMCell, DropoutWrapper

class ESIM(object):
    
    def __init__(self, seq_length, n_vocab, embedding_size, hidden_size, attention_size, classes_num, batch_size, learning_rate, optimizer, l2, clip_value):
        self._parameter_init(seq_length, n_vocab, embedding_size, hidden_size, attention_size, batch_size, learning_rate, optimizer, l2, clip_value)
        self._placeholder_init()
        self._y_pred = self._forward()
        self._loss = self._getloss()
        self._acc = self._getAcc()
        self._train = self._train()

        tf.add_to_collection('train_mini', self.train)

    def _parameter_init(self, seq_length, n_vocab, embedding_size, hidden_size, attention_size, classes_num, batch_size, learning_rate, optimizer, l2, clip_value):
        self._seq_length = seq_length
        self._n_vocab = n_vocab
        self._embedding_size = embedding_size
        self._hidden_size = hidden_size
        self._attention_size = attention_size
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._optimizer = optimizer
        self._l2 = l2
        self._clip_value = clip_value
        self._classes_num = classes_num

    def _placeholder_init(self):
        self._premise = tf.placeholder(tf.int32, [None, self._seq_length], 'premise') # equal to question_1 in DQD
        self._hypothesis = tf.placeholder(tf.int32, [None, self.seq_length], 'hypothesis')
        self._y = tf.placeholder(tf.float32, [None], 'y_true')
        self._premise_mask = tf.placeholder(tf.int32, [None], 'premise_actual_length')
        self._hypothesis_mask = tf.placeholder(tf.int32, [None], 'hypothesis_actual_length')
        self._embedding_matrix = tf.placeholder(tf.float32,[self._n_vocab, self._embedding_size], 'embedding_matrix')
        self._dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

    def _forward(self):
        a_bar, b_bar = self._inputEncoding('input_encoding')
        m_a, m_b = self._localInference(a_bar, b_bar, 'local_inference')
        y_pred = self._inferenceComposition(m_a, m_b, 'inference_composition')
        return y_pred

    def _getloss(self, l2_lambda = 0.0001):
        with tf.name_scope('cost'):
            cross_entropy = -self._y * tf.log(self._y_pred) - (1 - self._y) * tf.log(1 - self._y_pred)
            loss = tf.reduce_mean(cross_entropy)
            weights = [v for v in tf.trainable_variables() if ('w' in v.name) or ('kernel' in v.name)]
            print(tf.trainable_variables())
            l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in weights]) * l2_lambda
            loss += l2_loss
        return loss
    
    def _getAcc(self):
        with tf.name_scope('acc'):
            label_pred = tf.round(self._y_pred, name='label_pred')
            correct_pred = tf.equal(tf.cast(label_pred, tf.int32), tf.cast(self._y, tf.int32))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='Accuracy')
        return accuracy

    def _train(self):
        with tf.name_scope('training'):
            if self.optimizer == 'adam':
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
            elif self.optimizer == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
            elif self.optimizer == 'momentum':
                optimizer = tf.train.MomentumOptimizer(self.learning_rate, momentum=0.9)
            elif self.optimizer == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            elif self.optimizer == 'adadelta':
                optimizer = tf.train.AdadeltaOptimizer(self.learning_rate)
            elif self.optimizer == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(self.learning_rate)
            else:
                ValueError('Unknown optimizer : {0}'.format(self.optimizer))
        gradients, v = zip(*optimizer.compute_gradients(self.loss))
        if self.clip_value is not None:
            gradients, _ = tf.clip_by_global_norm(gradients, self.clip_value)
        train_op = optimizer.apply_gradients(zip(gradients, v))
        return train_op


    def _inputEncoding(self, scope):
        # 3.1 Input Encoding
        # get sentence embedding
        with tf.device('/cpu:0'):
            self._Embedding = tf.get_variable('Embedding', [self.n_vocab, self.embedding_size], tf.float32)
            self._embedded_left = tf.nn.embedding_lookup(self._Embedding, self._premise)
            self._embedded_right = tf.nn.embedding_lookup(self._Embedding, self._hypothesis)
        
        # a¯i = BiLSTM(a, i), ∀i ∈ [1, . . . , `a], (1)
        # b¯j = BiLSTM(b, j), ∀j ∈ [1, . . . , `b]. (2)
        with tf.variable_scope(scope):
            output_premise, final_state_premise = self._biLSTM(self._embedded_left,  self._hidden_size, 'biLstm', self._premise_mask)
            output_hypothesis, final_state_hypothesis = self._biLSTM(self._embedded_right,  self._hidden_size, 'biLstm', self._hypothesis_mask, True)
            a_bar = tf.concat(output_premise, axis=2)
            b_bar = tf.concat(output_hypothesis, axis=2)
            return a_bar, b_bar

    def _localInference(self, a_bar, b_bar,scope):
        # 3.2 Local Inference Modeling
        with tf.variable_scope(scope):
            attention_weight = tf.matmul(a_bar, tf.transpose(b_bar, [0, 2, 1]))
            # softmax
            attention_soft_a = tf.nn.softmax(attention_weight)
            attention_soft_b = tf.nn.softmax(tf.transpose(attention_weight, [0, 2, 1]))

            a_hat = tf.matmul(attention_soft_a, b_bar)
            b_hat = tf.matmul(attention_soft_b, a_bar)

            # m_a = [a_bar, a_hat, a_bar - a_hat, a_bar 'dot' a_hat] (14)
            # m_b = [b_bar, b_hat, b_bar - b_hat, b_bar 'dot' b_hat] (15)

            a_diff = tf.subtract(a_bar, a_hat)
            b_diff = tf.subtract(b_bar, b_hat)

            a_mul = tf.multiply(a_bar, a_hat)
            b_mul = tf.multiply(b_bar, b_hat)

            m_a = tf.concat([a_bar, a_hat, a_diff, a_mul], axis = 2)
            m_b = tf.concat([b_bar, b_hat, b_diff, b_mul], axis = 2)

            return m_a, m_b

    def _inferenceComposition(self, m_a, m_b, scope):
        # 3.3 inferenceComposition
        with tf.variable_scope(scope):
            outputV_a, finalStateV_a = self._biLSTM(m_a, self._hidden_size, 'biLSTM')
            outputV_b, finalStateV_b = self._biLSTM(m_b, self._hidden_size, 'biLSTM', isReuse = True)
            v_a = tf.concat(outputV_a, axis = 2) # (batch_size, seq_length, 2 * hidden_size)
            v_b = tf.concat(outputV_b, axis = 2)

            # get average
            v_a_avg = tf.reduce_mean(v_a, axis = 1) # (batch_size, 2 * hidden_size)
            v_b_avg = tf.reduce_mean(v_b, axis = 1)

            # get max
            v_a_max = tf.reduce_max(v_a, axis = 1) # (batch_size, 2 * hidden_size)
            v_b_max = tf.reduce_max(v_b, axis = 1)
            
            # v = [v_{a,avg}; v_{a,max}; v_{b,avg}; v_{b_max}] (20)
            v = tf.concat([v_a_avg, v_a_max, v_b_avg, v_b_max], axis = 1)

            y_hat = self._finalMultilayer(v, self._hidden_size, self._classes_num, 'final_multilayer')
            return y_hat

    def  _finalMultilayer(self, inputs, hidden_dims, num_units, scope, isReuse = False, initializer = None):
        with tf.variable_scope(scope):
            if initializer is None:
                initializer = tf.random_normal_initializer(0.0, 0.1)

            with tf.variable_scope('hidden_layer1'):
                inputs = tf.nn.dropout(inputs, self._dropout_keep_prob)
                hidden_outputs = tf.layer.dense(inputs, hidden_dims, tf.nn.relu, kernel_initializer = initializer)
            
            with tf.variable_scope('output_layer2'):
                hidden_outputs = tf.nn.dropout(hidden_outputs, self._dropout_keep_prob)
                result = tf.layer.dense(hidden_outputs, num_units, tf.nn.sigmoid, kernel_initializer = initializer)
                return result

    def _biLSTM(self, inputs, num_units, scope, seq_length = None, isReuse = False):
        with tf.variable_scope(scope, reuse = isReuse):
            lstm_cell = LSTMCell(num_units = num_units)
            forward_lstm_cell = DropoutWrapper(lstm_cell, output_keep_prob = self._dropout_keep_prob)
            backward_lstm_cell = DropoutWrapper(lstm_cell, output_keep_prob = self._dropout_keep_prob)
            output = tf.nn.bidirectional_dynamic_rnn(cell_fw = forward_lstm_cell,
                                                     cell_bw = backward_lstm_cell,
                                                     inputs = inputs,
                                                     sequence_length = seq_length,
                                                     dtype = tf.float32)
            return output

