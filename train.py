#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import logging
import os

import tensorflow as tf

import utils
from model import Model
from utils import read_data
from utils import build_dataset
from utils import get_train_data

from flags import parse_args
FLAGS, unparsed = parse_args()

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s', level=logging.DEBUG)

# 读入数据,返回的保存所有文字,标点符号的list
vocabulary = read_data(FLAGS.text)

print('Data size', len(vocabulary))

vocabulary_size =5000

# 获得每个文字对应的编号字典dictionary{文字:编号}; reversed_dictionary{编号:文字}
# 同时会在当前路径保存这两个字典的json文件:dictionary.json;reversed_dictionary.json
_, _, dictionary, reversed_dictionary = build_dataset(vocabulary,vocabulary_size)

model = Model(learning_rate=FLAGS.learning_rate, batch_size=FLAGS.batch_size, num_words=vocabulary_size, num_steps=FLAGS.num_steps)

model.build()

with tf.Session() as sess:
    summary_string_writer = tf.summary.FileWriter(FLAGS.output_dir, sess.graph)

    saver = tf.train.Saver(max_to_keep=5)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    logging.debug('Initialized')

    try:
        logging.debug('FLAGS.output_dir [{0}]'.format(FLAGS.output_dir))
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.output_dir)
        logging.debug('checkpoint_path [{0}]'.format(checkpoint_path))
        saver.restore(sess, checkpoint_path)
        logging.debug('restore from [{0}]'.format(checkpoint_path))

    except Exception:
        logging.debug('no check point found....')

    print("vocabulary:",len(vocabulary))
    print("batch_size:",FLAGS.batch_size)
    print("len_batch:",len(vocabulary)//FLAGS.batch_size)
    print("num_steps:",FLAGS.num_steps)

    epoch_size=len(vocabulary)//FLAGS.batch_size//FLAGS.num_steps

    print("epoch_size:",epoch_size)

    for x in range(1):
        logging.debug('epoch [{0}]....'.format(x))

        state = sess.run(model.init_state)
      
        for dl in get_train_data(vocabulary, dictionary, batch_size=FLAGS.batch_size, num_steps=FLAGS.num_steps):
            
            ##################
            # Your Code hereXX
            ##################
            # print("dl[0]",dl[0].shape)
            # print("dl[1]",dl[1].shape)
            feed_dict = {
                        model.X:dl[0], 
                        model.Y:dl[1],
                        model.keep_prob:0.9, 
                        model.init_state:state}
      
            grobal_step, _, state, loss, summary_string = sess.run([model.global_step,
                                                        model.optimizer, 
                                                        model.final_state, 
                                                        model.loss, 
                                                        model.merged_summary_op], feed_dict=feed_dict)
            
            # print("summary_string",summary_string)
            summary_string_writer.add_summary(summary_string, grobal_step)
            # print("loss",loss)

            if grobal_step % 10 == 0:
                logging.debug('step [{0} / {1}] loss [{2}]'.format(grobal_step, epoch_size, loss))
                save_path = saver.save(sess, os.path.join(
                    FLAGS.output_dir, "model.ckpt"), global_step=grobal_step)
            
    summary_string_writer.close()
