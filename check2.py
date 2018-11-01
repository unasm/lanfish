# coding: utf-8
import time
from datetime import timedelta

import numpy as np

import sys
import csv
import json
from math import log
import os
import jieba
import jieba.analyse
import nltk
from sklearn import metrics

import tensorflow.contrib.keras as kr

reload(sys)
sys.setdefaultencoding("utf8")
from tnn import TCNNConfig, TCNN, TextRNN
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
        subtags = []
        try:
            tag_arr = line[1].split("'")
            for x in tag_arr:
                x = x.strip().strip(",")
                if x == "" or x == "[" or x == "]" or x == "[]":
                    continue
                tag = native_content(x.strip("'"))
                subtags.append(tag)
        except Exception as ex:
            print(ex)
        contents.append(native_content(line[0]))
        labelList.append(label)
        tags.append(subtags)
        if label in labelDict:
            continue
        # 给每个label编号
        labelDict[label] = labelIdx
        #label名称的列表
        labelNames.append(label)
        labelIdx += 1
    return contents, tags, labelList, labelDict, labelNames
    #for line in open_file(filePath):
    #    print(line)
    #    print(len(line.split(",")))

def cutWords(dirFile, contents):
    stopDict = getStopDict(stopFile)
    for content in contents:
        # if cnt <= 0:
        #    break
        # cnt -= 1
        # jieba.cut(content)
        seg_list = jieba.cut(content, cut_all=True)
        subwords = []
        for seg in seg_list:
            if seg not in stopDict and len(seg) > 1:
                #过滤掉停用
                #print(seg)
                subwords.append(seg)

        # content = "＊＊＊    式（Ⅰ）、（Ⅱ）、（Ⅲ）化合物。"
        # content = "本发明涉及３－（２－氰基苯基）－５－（２－吡啶基）－１－苯基－１，２－二氢吡啶－２－酮的无定形物。"
        #seg_list = jieba.analyse.textrank(content, topK=45, withWeight=True, allowPOS=('ns', 'n', 'vn', 'v'))
        print("len of sub_words : ", len(subwords))
        try:
            if len(subwords) > 0:
                open_file(dirFile, mode='a').write(','.join(subwords) + '\n')
            else:
                open_file(dirFile, mode='a').write(",\n")
        except Exception as ex:
            print(ex)
            print(seg_list)
            print(content)

def build_vocab(wordsFilePath, tags, vocab_dir):
    """根据训练集构建词汇表，存储, 把最常见的5000个取出来"""


    words = []
    is_tag_count = False

    wordCnt = {}
    all_seg_list = []
    print("len_of_contents", len(contents))
    for wordList in open(wordsFilePath, "r"):
        #if cnt <= 0:
        #    break
        #cnt -= 1
        # jieba.cut(content)
        # seg_list = jieba.cut(content, cut_all=False)
        #content = "＊＊＊    式（Ⅰ）、（Ⅱ）、（Ⅲ）化合物。"
        #content = "本发明涉及３－（２－氰基苯基）－５－（２－吡啶基）－１－苯基－１，２－二氢吡啶－２－酮的无定形物。"
        #seg_list = jieba.analyse.textrank(content, topK=15, withWeight=True, allowPOS=('ns', 'n', 'vn', 'v'))
        #subwords = []
        #if len(seg_list) == 0:
        #    seg_list = jieba.cut(content, cut_all=False)
        #    for seg in seg_list:
        #        if len(seg) > 1:
        #            subwords.append(seg)
        #else:
        #    subwords = zip(*seg_list)[0]
        #try:
        #    if len(subwords) > 0:
        #        open_file(contentSplitFile, mode='a').write(','.join(subwords) + '\n')
        #    else:
        #        open_file(contentSplitFile, mode='a').write(',\n')
        #except Exception as ex:
        #    print(ex)
        #    print(seg_list)
        #    print(content)
        subwords = wordList.split(",")

        for seg in subwords:
            if seg == "":
                continue
            # print(native_content(seg[0]))
            seglocal = native_content(seg)
            all_seg_list.append(seglocal)
                # print(seglocal)
    counter = Counter(all_seg_list)
    count_pairs = counter.most_common(config.vocab_size - 1)
    print("select_words", len(count_pairs))
    for word in count_pairs:
        words.append(word[0])


    if is_tag_count:
        tagWords = []
        tagWordDict = {}
        tagCnt = 0
        #allTag = 0
        for tag in tags:
            for word in tag:
                #allTag += 1
                if word in tagWordDict:
                    continue
                tagCnt += 1
                tagWordDict[word] = True
                tagWords.append(word)

        #all_data = []
        #for content in contents:
        #    all_data.extend(content)
        #    # all_data.extend(content.decode("utf-8"))
        #counter = Counter(all_data)
        #count_left = config.vocab_size - 1 - tagCnt
        #print("count left ", count_left)

        #count_pairs = counter.most_common(config.vocab_size - 1)
        #wordsTuple, _ = list(zip(*count_pairs))

        #print("sizeof_word_tuple ", len(wordsTuple))
        #for word in wordsTuple:
        #    words.append(word)



    #for word in tagWords:
    #    words.append(word)

    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<PAD>'] + list(words)
    print("words_of length", len(words))
    open_file(vocab_dir, mode='w').write('\n'.join(words) + '\n')


def read_vocab(vocab_dir):
    """读取词汇表"""
    # words = open_file(vocab_dir).read().strip().split('\n')
    with open_file(vocab_dir) as fp:
        # 如果是py2 则每个值都转化为unicode
        words = [native_content(_.strip()) for _ in fp.readlines()]
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id

def getLabelFeature(splitFilePath, labels):
    lineCnt = 0
    labelWords = {}
    for line in open(splitFilePath, "r"):
        #print(line)
        label = labels[lineCnt]
        if label not in labelWords:
            labelWords[label] = []
        lineCnt += 1
        lineArr = line.split(",")
        for word in lineArr:
            if word != "":
                labelWords[label].append(word)

    feature_each_label = {}
    for label in labelWords:
        counter = Counter(labelWords[label])
        count_pairs = counter.most_common(config.words_of_each_label)
        feature_each_label[label] = {}
        for word in count_pairs:
            feature_each_label[label][word[0]] = True
    return feature_each_label


def process_file(word_to_id, label_to_id, splitFilePath, labels, tags, features, max_length=2700):
    data_ids = []
    label_ids = []
    i = 0
    #feature_each_label = getLabelFeature(splitFilePath, labels)
    maxlen_of_words = 0
    for line in open(splitFilePath, "r"):
        wordlist = []
        label = labels[i]
        #for tag in tags[i]:
        #    # tags 优先放入队列
        #    if tag in word_to_id:
        #        wordlist.append(word_to_id[tag])
        #for x in contents[i]:
        splitWords = line.split(",")
        setWords = set(splitWords)
        for feature in features:
            if feature in setWords:
                wordlist.append(word_to_id[feature])
            #else:
            #    wordlist.append(0)
        if len(wordlist) > maxlen_of_words:
            maxlen_of_words = len(wordlist)
            print("maxlen_of_words : ", maxlen_of_words)
        data_ids.append(wordlist)
        try:
            if labels[i] not in label_to_id:
                print("label not in")
                print(labels[i])
                continue
        except Exception as ex:
            print(ex)
            print(i)
            print(len(labels))
            print(type(labels))
            print(labels[i])
        label_ids.append(label_to_id[labels[i]])
        i += 1
    x_pad = kr.preprocessing.sequence.pad_sequences(data_ids, config.seq_length)
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

def calculate_B_from_A(A):
    '''
    :param A: CHI公式中的A值
    :return: B，CHI公职中的B值。不是某一类但是也包含单词t的文档。
    '''
    B = {}
    for key in A:
        B[key] = {}
        for word in A[key]:
            B[key][word] = 0
            for kk in A:
                if kk != key and A[kk].has_key(word):
                    #B[key][word] += 1
                    B[key][word] += A[kk][word]
    return B


def feature_select_use_new_CHI(A, B, count):
    '''
    根据A，B，C，D和CHI计算公式来计算所有单词的CHI值，以此作为特征选择的依据。
    CHI公式：chi = N*（AD-BC）^2/((A+C)*(B+D)*(A+B)*(C+D))其中N,(A+C),(B+D)都是常数可以省去。
    :param A:
    :param B:
    :return: 返回选择出的1000多维特征列表。
    '''
    word_features = []
    #for i in range(0, 11):
    print("count_of_labels : ", len(count))
    for label in count:
        CHI = {}
        M = config.N - count[label]
        for word in A[label]:
            #print word, A[i][word], B[i][word]
            temp = (A[label][word] * (M - B[label][word]) - (count[label] - A[label][word]) * B[label][word]) ^ 2 / (
            (A[label][word] + B[label][word]) * (config.N - A[label][word] - B[label][word]))
            value = config.N / (A[label][word] + B[label][word])
            CHI[word] = log(value) * temp
        #每一类新闻中只选出150个CHI最大的单词作为特征
        a = sorted(CHI.iteritems(), key=lambda t: t[1], reverse=True)[:200]
        #b = []
        #for aa in a:
        #    b.append(aa[0])
        #word_dict.extend(b)
        for word in a:
            if word[0] not in word_features:
                word_features.append(word[0])
    print("len of word_features", len(word_features))
    return word_features

def getFenci(fenciSplitPath):
    resList = []
    for line in open(fenciSplitPath, "r"):
        words = line.split(",")
        resList.append((words, set(words)))
    return resList

def document_features(word_features, data):
    '''
    计算每一篇新闻的特征向量权重。即将文件从分词列表转化为分类器可以识别的特征向量输入。
    :param word_features:
    :param TFIDF:
    :param document: 分词列表。存储在train_set,test_set中
    :param cla: 类别
    :param num: 文件编号
    :return: 返回该文件的特征向量权重
    '''
    document_words = set(data)
    features = {}
    for i, word in enumerate(word_features):
        if word in document_words:
            features[word] = 1#TF[num][word]#*log(N/(A[cla][word]+B[cla][word]))
        else:
            features[word] = 0
    return features


def getWordsCount(trainSplitFile, trainLabels, testSplitFile, testLabels):
    '''
    本函数用于处理样本集中的所有文件。并返回处理结果所得到的变量
    :param floder_path: 样本集路径
    :return: A：CHI公示中的A值，嵌套字典。用于记录某一类中包含单词t的文档总数。第一层总共9个key，对应9类新闻分类
                第二层则是某一类中所有单词及其包含该单词的文档数（而不是出现次数）。{{1：{‘hello’：8，‘hai’：7}}，{2：{‘apple’：8}}}
            TFIDF：用于计算TFIDF权值。三层嵌套字典。第一层和A一样，key为类别。第二层的key为文件名（这里使用文件编号代替0-99）.第三层
                    key为单词，value为盖单词在本文件中出现的次数。用于记录每个单词在每个文件中出现的次数。
            train_set:训练样本集。与测试样本集按7:3比例分开。三元组（文档的单词表，类别，文件编号）
            test_set:测试样本集。三元组（文档的单词表，类别，文件编号）
    '''
    # 用于记录CHI公示中的A值
    A = {}
    tf = []
    i=0
    # 存储训练集/测试集
    count = {}
    train_set = []
    test_set = []
    trainWordList = getFenci(trainSplitFile)

    for i in range(0, len(contents)):
        tf.append({})
        label = trainLabels[i]
        if label not in A:
            A[label] = {}
        if label not in count:
            count[label] = 0
        count[label] += 1
        #word_list, word_set = getFenci(content)
        train_set.append((trainWordList[i][0], label))
        for word in trainWordList[i][1]:
            if A[label].has_key(word):
                A[label][word] += 1
            else:
                A[label][word] = 1
        for word in trainWordList[i][1]:
            if tf[i].has_key(word):
                tf[i][word] += 1
            else:
                tf[i][word] = 1

    tf2 = []
    testWordList = getFenci(testSplitFile)
    for i in range(0, len(testWordList)):
        tf2.append({})
        label = testLabels[i]
        test_set.append((testWordList[i][0], label))
        for word in testWordList[i][1]:
            if tf2[i].has_key(word):
                tf2[i][word] += 1
            else:
                tf2[i][word] = 1
    print "处理完数据"
    return A, tf, tf2, train_set, test_set, count

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


def getStopDict(stopFile):
    stopDict = {}
    for word in open(stopFile, "r"):
        stopDict[native_content(word.strip())] = True
    return stopDict



if __name__ == "__main__":
    trainFilePath = "./data/task_dataset_train.csv"
    testFilePath = "./data/task_dataset_test.csv"
    stopFile = "data/stopWords.txt"
    trainContentSplitFile = "data/traincontentcut_2.txt"
    testContentSplitFile = "data/testcontentcut_2.txt"
    featureWordFile = "data/feature_words_2.txt"
    vocab_file = "./data/vocab.txt"


    config = TCNNConfig()
    contents, tags, labelList, labels_to_id, labelNames = getData(trainFilePath)
    test_contents, test_tags, testLabelList, test_labels_to_id, test_labelNames = getData(testFilePath)
    if not os.path.exists(trainContentSplitFile):
        cutWords(trainContentSplitFile, contents)
    if not os.path.exists(testContentSplitFile):
        cutWords(testContentSplitFile, test_contents)
    #if not os.path.exists(vocab_file):
    #    build_vocab(trainContentSplitFile, tags, vocab_file)

    A, tf_res, tf2_res, train_set, test_set, count = getWordsCount(trainContentSplitFile, labelList, testContentSplitFile, testLabelList)
    B = calculate_B_from_A(A)
    word_features = feature_select_use_new_CHI(A, B, count)

    #json.dump(word_features, open(featureWordFile, 'w'))
    #if not os.path.exists(featureWordFile):
    #    B = calculate_B_from_A(A)
    #    word_features = feature_select_use_new_CHI(A, B, count)
    #    json.dump(word_features, open(featureWordFile, 'w'))
    #else:
    #    word_features = json.load(open(featureWordFile, 'r'))
    #for i in range(0, len(word_features)):
    #    word_features[i] = native_content(word_features[i])

    documents_feature = [(document_features(word_features, data[0]), data[1]) for i, data in enumerate(train_set)]

    test_documents_feature = [(document_features(word_features, data[0]), data[1]) for i, data in enumerate(test_set)]

    classifier = nltk.NaiveBayesClassifier.train(documents_feature)
    classifier.show_most_informative_features(20)
    print("train_error : ", nltk.classify.accuracy(classifier, documents_feature))
    print("test_error : ", nltk.classify.accuracy(classifier, test_documents_feature))

    #words, word_to_id = read_vocab(vocab_file)
    #print("len of features : ", len(word_features))
    #word_to_id = dict(zip(word_features, range(len(word_features))))

    #train_x_pad, train_y_pad = process_file(word_to_id, labels_to_id, trainContentSplitFile, labelList, tags, word_features, config.vocab_size)
    #test_x_pad, test_y_pad = process_file(word_to_id, labels_to_id, testContentSplitFile, testLabelList, tags, word_features, config.vocab_size)
    #config = TCNNConfig()
    ###model = TextRNN(config)
    #model = TCNN(config)
    #if sys.argv[1] == "distbute":
    #    labelCnt = {}
    #    for label in labelList:
    #        print(label)
    #        if label not in labelCnt:
    #            labelCnt[label] = 0
    #        labelCnt[label] += 1
    #    for label in labelCnt:
    #        print("label is : %s \t count is : %d" % (label, labelCnt[label]))
    #elif sys.argv[1] == "train":
    #    train(train_x_pad, train_y_pad, test_x_pad, test_y_pad)
    #else:
    #    test(test_x_pad, test_y_pad)
