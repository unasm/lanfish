# -*- coding: utf-8 -*-

import tensorflow as tf

class TCNNConfig(object):
    embedding_dim = 128  # 词向量维度
    seq_length = 100    # 序列长度
    num_classes = 49    # 类别数
    num_filters = 256   # 卷积核 核数
    kernel_size  = 3    #核的大小

    #vocab_size = 35000   #词表的大小

    vocab_size = 4910   #词表的大小
    #vocab_size = 2700   #词表的大小
    hidden_dim = 128  # 全连接层神经元
    hidden_dim = 128    #全联接层的神经元

    dropout_keep_prob = 0.9  #drop out的比例
    #learning_rate = 0.01
    learning_rate = 0.001
    batch_size = 64         #每次送入的数据大小
    num_epochs = 1000       #  总迭代轮次, 总迭代轮次
    print_per_batch = 100   # 迭代100次输出一次过程
    save_per_batch = 10     # 每迭代多少次送入一次到 tensorboard
    item_max_cnt = 4000
    words_of_each_label = 100
    N = 16571


class TCNN(object):

    def __init__(self, config):
        self.config = config
        self.input_x = tf.placeholder(tf.int32, [None, config.seq_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, config.num_classes], name="input_y")
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        self.cnn()

    def cnn(self):
        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [self.config.vocab_size, self.config.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)

        with tf.name_scope("cnn"):
            conv = tf.layers.conv1d(embedding_inputs, self.config.num_filters, self.config.kernel_size, name="conv")
            gmp = tf.reduce_max(conv, reduction_indices=[1], name="gmp")

        with tf.name_scope("score"):
            fc = tf.layers.dense(gmp, self.config.hidden_dim, name="fc1")
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            #fc = tf.nn.relu6(fc)
            fc = tf.nn.relu(fc)

            self.logits = tf.layers.dense(fc, self.config.num_classes, name="fc2")
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)

        #交叉熵
        with tf.name_scope("optimize"):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            #self.loss = tf.reduce_sum(cross_entropy)
            self.loss = tf.reduce_mean(cross_entropy)
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

class TextRNN(object):
    """文本分类，RNN模型"""
    def __init__(self, config):
        self.config = config

        # 三个待输入的数据
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.rnn()

    def rnn(self):
        """rnn模型"""

        def lstm_cell():   # lstm核
            return tf.contrib.rnn.BasicLSTMCell(self.config.hidden_dim, state_is_tuple=True)

        def gru_cell():  # gru核
            return tf.contrib.rnn.GRUCell(self.config.hidden_dim)

        def dropout(): # 为每一个rnn核后面加一个dropout层
            if (self.config.rnn == 'lstm'):
                cell = lstm_cell()
            else:
                cell = gru_cell()
            return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

        # 词向量映射
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)

        with tf.name_scope("rnn"):
            # 多层rnn网络
            cells = [dropout() for _ in range(self.config.num_layers)]
            rnn_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

            _outputs, _ = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=embedding_inputs, dtype=tf.float32)
            last = _outputs[:, -1, :]  # 取最后一个时序输出作为结果

        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(last, self.config.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)

            # 分类器
            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
