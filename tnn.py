# -*- coding: utf-8 -*-

import tensorflow as tf

class TCNNConfig(object):
    embedding_dim = 64  # 词向量维度
    seq_length = 600    # 序列长度
    num_classes = 49    # 类别数
    num_filters = 256   # 卷积核 核数
    kernel_size  = 5    #核的大小
    vocab_size = 6000   #词表的大小
    hidden_dim = 128  # 全连接层神经元
    hidden_dim = 128    #全联接层的神经元

    dropout_keep_prob = 0.5  #drop out的比例
    learning_rate = 0.001
    batch_size = 64         #每次送入的数据大小
    num_epochs = 1000       #  总迭代轮次, 总迭代轮次
    print_per_batch = 100   # 迭代100次输出一次过程
    save_per_batch = 10     # 每迭代多少次送入一次到 tensorboard
    item_max_cnt = 400


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
            fc = tf.nn.relu(fc)

            self.logits = tf.layers.dense(fc, self.config.num_classes, name="fc2")
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)

        with tf.name_scope("optimize"):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

