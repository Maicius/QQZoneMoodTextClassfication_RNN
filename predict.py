# coding: utf-8

from __future__ import print_function

import os
import tensorflow as tf
import tensorflow.contrib.keras as kr
import pandas as pd
from cnn_model import TCNNConfig, TextCNN
from data.cnews_loader import read_category, read_vocab
from rnn_model import TextRNN, TRNNConfig
try:
    bool(type(unicode))
except NameError:
    unicode = str

base_dir = 'data/qqzone'
vocab_dir = os.path.join(base_dir, 'text_vocab.txt')

save_dir = 'checkpoints/text_rnn'
save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径


class CnnModel:
    def __init__(self):
        self.config = TCNNConfig()
        self.categories, self.cat_to_id = read_category()
        self.words, self.word_to_id = read_vocab(vocab_dir)
        self.config.vocab_size = len(self.words)
        self.model = TextCNN(self.config)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=self.session, save_path=save_path)  # 读取保存的模型

    def predict(self, message):
        # 支持不论在python2还是python3下训练的模型都可以在2或者3的环境下运行
        content = unicode(message)
        data = [self.word_to_id[x] for x in content if x in self.word_to_id]

        feed_dict = {
            self.model.input_x: kr.preprocessing.sequence.pad_sequences([data], self.config.seq_length),
            self.model.keep_prob: 1.0
        }

        y_pred_cls = self.session.run(self.model.y_pred_cls, feed_dict=feed_dict)
        return self.categories[y_pred_cls[0]]

class RnnModel:
    def __init__(self):
        self.config = TRNNConfig()
        self.categories, self.cat_to_id = read_category()
        self.words, self.word_to_id = read_vocab(vocab_dir)
        self.config.vocab_size = len(self.words)
        self.model = TextRNN(self.config)
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=self.session, save_path=save_path)  # 读取保存的模型

    def predict(self, message):
        # 支持不论在python2还是python3下训练的模型都可以在2或者3的环境下运行
        content = unicode(message)
        data = [self.word_to_id[x] for x in content if x in self.word_to_id]

        feed_dict = {
            self.model.input_x: kr.preprocessing.sequence.pad_sequences([data], self.config.seq_length),
            self.model.keep_prob: 1.0
        }

        y_pred_cls = self.session.run(self.model.y_pred_cls, feed_dict=feed_dict)
        return self.categories[y_pred_cls[0]]

if __name__ == '__main__':
    cnn_model = RnnModel()
    data = pd.read_csv('/Users/maicius/code/InterestingCrawler/QQZone/data/train/text_maicius.csv', sep='\t')
    content = data['content'].values
    test_demo = ['团日活动，今天你吃撑了吗']
    for i in range(len(content)):
        label = cnn_model.predict(content[i])
        data.loc[i, 'Y'] = label
    data.to_csv('/Users/maicius/code/InterestingCrawler/QQZone/data/train/text_maicius_label.csv')
