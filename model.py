# #!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import logging 

class Model():
    def __init__(self, learning_rate=0.001, batch_size=16, num_steps=32, num_words=5000, dim_embedding=128, rnn_layers=3):
        r"""初始化函数

        Parameters
        ----------
        learning_rate : float 0.001
            学习率.
        batch_size : int 16 
            batch_size.
        num_steps : int 32
            RNN有多少个time step，也就是输入数据的长度是多少.
        num_words : int 5000
            字典里有多少个字，用作embeding变量的第一个维度的确定和onehot编码.
        dim_embedding : int 128
            embding中，编码后的字向量的维度
        rnn_layers : int 3
            有多少个RNN层，在这个模型里，一个RNN层就是一个RNN Cell，各个Cell之间通过TensorFlow提供的多层RNNAPI（MultiRNNCell等）组织到一起
            
        """
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.num_words = num_words
        self.dim_embedding = dim_embedding
        self.rnn_layers = rnn_layers
        self.learning_rate = learning_rate

    def build(self, embedding_file=None):
        # global step
        self.global_step = tf.Variable(
            0, trainable=False, name='self.global_step', dtype=tf.int64)

        self.X = tf.placeholder(
            tf.int32, shape=[None, self.num_steps], name='input')
        self.Y = tf.placeholder(
            tf.int32, shape=[None, self.num_steps], name='label')

        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        with tf.variable_scope('embedding'):
            if embedding_file:
                # if embedding file provided, use it.
                embedding = np.load(embedding_file)
                embed = tf.constant(embedding, name='embedding')
                self.lstm_inputs = tf.nn.embedding_lookup(embed, self.X)
                print("lstm_inputs:",self.lstm_inputs.shape)
            else:
                # if not, initialize an embedding and train it.
                with tf.device("/cpu:0"):
                    embed = tf.get_variable('embedding', [self.num_words, self.dim_embedding])
                    # tf.summary.histogram('embed', embed)
                    print("embed:",embed.shape)
                    self.lstm_inputs = tf.nn.embedding_lookup(embed, self.X)
                    print("lstm_inputs:",self.lstm_inputs.shape)

        with tf.variable_scope('rnn'):

            # 基础cell 也可以选择其他基本cell类型
            def lstm_cell():
                lstm = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.dim_embedding,forget_bias=0, state_is_tuple=True)
                drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=self.keep_prob)
                return drop

            # 多层cell 前一层cell作为后一层cell的输入 ,一共rnn_layers层
            cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell() for _ in range(self.rnn_layers)] ,state_is_tuple=True)
         
        
            # 初始状态生成(h0) 默认为0
            self.init_state = cell.zero_state(self.batch_size, tf.float32)

            # 将训练文本独热编码
            # lstm_inputs = tf.one_hot(self.X, self.num_words)

            # 使用dynamic_rnn自动进行时间维度推进 且 可以使用不同长度的时间维度
            # 'outputs' is a tensor of shape [max_time, batch_size, depth] ,if time_major=false
            # 'state' is a N-tuple where N is the number of LSTMCells containing a tf.contrib.rnn.LSTMStateTuple for each cell

            # state = self.init_state
            lstm_outputs_tensor, state = tf.nn.dynamic_rnn(cell, inputs=self.lstm_inputs, initial_state=self.init_state, time_major=False)
           
            final_outputs_tensor = lstm_outputs_tensor[:, -1, :]
            self.outputs_state_tensor = final_outputs_tensor
            self.final_state = state

        # print("lstm_outputs_tensor:", lstm_outputs_tensor.shape)
        # 沿着batch_size展开
        seq_output = tf.concat(lstm_outputs_tensor, 1)
        # print("seq_output:",seq_output.shape)

        # flatten it
        seq_output_final = tf.reshape(seq_output, [-1, self.dim_embedding])
        # print("seq_output_final:",seq_output_final.shape)

        with tf.variable_scope('softmax'):
            softmax_w = tf.get_variable("softmax_w", 
                                        [self.dim_embedding, self.num_words],
                                        initializer=tf.random_normal_initializer(stddev=0.01))
            softmax_b = tf.get_variable("softmax_b", 
                                        [self.num_words],
                                        initializer=tf.constant_initializer(0.0))

            

            # print("softmax_w:",softmax_w.shape)
            # print("softmax_b:",softmax_b.shape)
            logits = tf.matmul(seq_output_final, softmax_w) + softmax_b
            # print("logits:",logits.shape)

        tf.summary.histogram('logits', logits)

        self.predictions = tf.nn.softmax(logits, name='predictions')
        print("logits:",logits.shape)
        # 对标签进行独热编码
        y_one_hot = tf.one_hot(self.Y, self.num_words)
        print("y_one_hot:",y_one_hot.shape)
        labels = tf.reshape(y_one_hot, logits.get_shape())

        # print("labels:",labels.shape)
        
        # 使用tf.nn.sparse_softmax_cross_entropy_with_logits计算交叉熵
        # logits: 神经网络的最后一层输出，如果有batch的话，它的大小为[batch_size, num_classes], 单样本的话大小就是num_classes
        # labels: 样本的实际标签，大小[batch_size]。且必须采用labels=y_，logits=y的形式将参数传入。
        
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels)

        # 这个函数返回值不是一个数，而是一个向量，如果要求交叉熵，我们要在做一步tf.resuce_sum操作，
        # 如果要求loss，则需要做一步tf.reduce_mean操作，对向量求均值
         
        self.loss = tf.reduce_mean(loss)

        tf.summary.scalar('logits_loss', self.loss)

        _,variance = tf.nn.moments(logits, -1)
        var_loss = tf.divide(10.0, 1.0+tf.reduce_mean(variance))

        tf.summary.scalar('var_loss', var_loss)
        # 把标准差作为loss添加到最终的loss里面，避免网络每次输出的语句都是机械的重复
        self.loss = self.loss + var_loss
        tf.summary.scalar('total_loss', self.loss)
        
        # gradient clip = 5
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 5)

        train_op = tf.train.AdamOptimizer(self.learning_rate)
        
        self.optimizer = train_op.apply_gradients(
            zip(grads, tvars), global_step=self.global_step)

        tf.summary.scalar('loss', self.loss)
        
        self.merged_summary_op = tf.summary.merge_all()