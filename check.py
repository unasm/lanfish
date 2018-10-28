# coding: utf-8
import time
from datetime import timedelta

import numpy as np

import sys
import csv
import os
from sklearn import metrics

import tensorflow.contrib.keras as kr

reload(sys)
sys.setdefaultencoding("utf8")
from tnn import TCNNConfig, TCNN
from collections import Counter

import tensorflow as tf

save_dir = "checkpoints/demo"

save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径


def native_content(content):
    return content.decode('utf-8')
    #if not is_py3:
    #    return content.decode('utf-8')
    #else:
    #    return content

if sys.version_info[0] > 2:
    is_py3 = True
else:
    reload(sys)
    sys.setdefaultencoding("utf-8")
    is_py3 = False

def open_file(filename, mode='r'):
    """
    常用文件操作，可在python2和python3间切换.
    mode: 'r' or 'w' for read or write
    """
    if is_py3:
        return open(filename, mode, encoding='utf-8', errors='ignore')
    else:
        return open(filename, mode)


def getData(filePath):
    fp = open_file(filePath, "r")
    skipRows = 1
    labelList = []
    tags = []
    contents = []
    labelDict = {}
    labelIdx = 0
    labelNames = []

    labelItemCnt = {}
    for line in csv.reader(fp):
        if skipRows > 0:
            skipRows -= 1
            continue
        label = native_content(line[2])
        if label not in labelItemCnt:
            labelItemCnt[label] = 0
        if labelItemCnt[label] > config.item_max_cnt:
            continue
        labelItemCnt[label] += 1
        tags.append(line[1])
        contents.append(native_content(line[0]))
        labelList.append(label)
        if label in labelDict:
            continue
        labelDict[label] = labelIdx
        labelNames.append(label)
        labelIdx += 1
    return contents, tags, labelList, labelDict, labelNames
    #for line in open_file(filePath):
    #    print(line)
    #    print(len(line.split(",")))


def build_vocab(contents, vocab_dir, vocab_size=6000):
    """根据训练集构建词汇表，存储, 把最常见的5000个取出来"""
    all_data = []
    for content in contents:
        all_data.extend(content)
        #all_data.extend(content.decode("utf-8"))
    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<PAD>'] + list(words)
    open_file(vocab_dir, mode='w').write('\n'.join(words) + '\n')


def read_vocab(vocab_dir):
    """读取词汇表"""
    # words = open_file(vocab_dir).read().strip().split('\n')
    with open_file(vocab_dir) as fp:
        # 如果是py2 则每个值都转化为unicode
        words = [native_content(_.strip()) for _ in fp.readlines()]
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id


def process_file(word_to_id, label_to_id, contents, labels, max_length=600):
    data_ids = []
    label_ids = []
    for i in range(len(contents)):
        wordlist = [word_to_id[x] for x in contents[i] if x in word_to_id]
        data_ids.append(wordlist)
        if labels[i] not in label_to_id:
            print("label not in")
            print(labels[i])
            continue
        label_ids.append(label_to_id[labels[i]])
    x_pad = kr.preprocessing.sequence.pad_sequences(data_ids, max_length)
    y_pad = kr.utils.to_categorical(label_ids, num_classes=len(label_to_id))
    return x_pad, y_pad


def batch_iter(x_train, y_train, batch_size):
    """
        随机选择batch_size 迭代
    :param x_train:
    :param y_train:
    :param batch_size:
    :return:
    """
    x_len = len(x_train)
    num_batch = int((x_len - 1) / batch_size) + 1
    indices = np.random.permutation(np.arange(x_len))
    x_shuffle = x_train[indices]
    y_shuffle = y_train[indices]
    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, x_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]


def feed_data(x_batch, y_batch, keep_prob):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.keep_prob: keep_prob
    }
    return feed_dict

def evaluate(sess, x_, y_):
    """评估在某一数据上的准确率和损失"""
    data_len = len(x_)
    batch_eval = batch_iter(x_, y_, 128)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = feed_data(x_batch, y_batch, 1.0)
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len

    return total_loss / data_len, total_acc / data_len


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def train(train_x_pad, train_y_pad, test_x_pad, test_y_pad):
    print("Configuration and save")
    tensorboard_dir = "tensorboard/lanternfish"
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)
    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    start_time = time.time()

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)
    total_batch = 0
    best_acc_val = 0.0
    require_improvement = 1000  # 如果超过1000轮未提升，提前结束训练
    flag = False
    for epoch in range(config.num_epochs):
        print("Epoch: ", epoch + 1)
        batch_train = batch_iter(train_x_pad, train_y_pad, config.batch_size)

        for x_batch, y_batch in batch_train:
            feed_dict = feed_data(x_batch, y_batch, config.dropout_keep_prob)
            if total_batch % config.print_per_batch == 0:
                merged_run = session.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(merged_run, total_batch)
            if total_batch % config.print_per_batch == 0:
                feed_dict[model.keep_prob] = 1.0
                loss_train, acc_train = session.run([model.loss, model.acc], feed_dict=feed_dict)
                loss_val, acc_val = evaluate(session, test_x_pad, test_y_pad)
                if acc_val > best_acc_val:
                    best_acc_val = acc_val
                    last_improved = total_batch
                    saver.save(sess=session, save_path=save_path)
                    improved_str = "*"
                else:
                    improved_str = ""

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                        + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))
                msg = "iter : "
            session.run(model.optim, feed_dict=feed_dict)
            total_batch += 1
            if total_batch - last_improved > require_improvement:
                print("no optimization for a long time, auto-stoping")
                flag = True
                break
        if flag:
            break

def test(x_test, y_test):
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)
    loss_test, acc_test = evaluate(session, x_test, y_test)
    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    print(msg.format(loss_test, acc_test))

    batch_size = 128
    data_len = len(x_test)

    config.batch_size
    num_batch = int((data_len - 1) / config.batch_size) + 1
    y_test_cls = np.argmax(y_test, 1)
    y_pred_cls = np.zeros(shape=len(x_test), dtype=np.int32)
    for i in range(num_batch):
        start_id = i * batch_size;
        end_id = min((i + 1) * batch_size, data_len)
        feed_dict = {
            model.input_x: x_test[start_id:end_id],
            model.keep_prob: 1.0
        }
        # 保存预测结果
        y_pred_cls[start_id:end_id] = session.run(model.y_pred_cls, feed_dict=feed_dict)

    print("Precision, Recal and F1-score")
    print(metrics.classification_report(y_test_cls, y_pred_cls, target_names=labelNames))




if __name__ == "__main__":
    trainFilePath = "/Users/unasm/project/lantern/text-classification-cnn-rnn/lanternfish/task_dataset_train.csv"
    testFilePath = "/Users/unasm/project/lantern/text-classification-cnn-rnn/lanternfish/task_dataset_test.csv"
    vocab_file = "/Users/unasm/project/lantern/text-classification-cnn-rnn/lanternfish/data/vocab.txt"

    config = TCNNConfig()
    contents, tags, labelList, labels_to_id, labelNames = getData(trainFilePath)
    test_contents, test_tags, testLabelList, test_labels_to_id, test_labelNames = getData(testFilePath)
    if not os.path.exists(vocab_file):
        build_vocab(contents, vocab_file)


    #labels_to_id = dict(zip(labels, range(len(labels))))
    words, word_to_id = read_vocab(vocab_file)
    train_x_pad, train_y_pad = process_file(word_to_id, labels_to_id, contents, labelList)
    test_x_pad, test_y_pad = process_file(word_to_id, labels_to_id, test_contents, testLabelList)
    config = TCNNConfig()
    model = TCNN(config)
    if sys.argv[1] == "distbute":
        labelCnt = {}
        for label in labelList:
            print(label)
            if label not in labelCnt:
                labelCnt[label] = 0
            labelCnt[label] += 1
        for label in labelCnt:
            print("label is : %s \t count is : %d" % (label, labelCnt[label]))
    elif sys.argv[1] == "train":
        train(train_x_pad, train_y_pad, test_x_pad, test_y_pad)
    else:
        test(test_x_pad, test_y_pad)
