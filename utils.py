#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import collections
import random
import json
import numpy as np

def fix_data(filename):
    #   读取中文宋词
    line_words = []
    lines1 = []

    with open(filename,"r",encoding="UTF-8") as f:
        line = f.readline()

        while line:
            while '\n' in line:
                line = line.replace('\n','') # 换行符替换

            lines1.append(line)

            line=f.readline()

    for i in range(len(lines1)-1):
        li1=lines1[i]
        li2=lines1[i+1]

        if len(li1) < 4:
            if len(li2) < 14:
                # lines1[i] 是作者
                pass
            else:
                # lines1[i] 是词牌名
                line_words.append(li1)
        else:
            # lines1[i] 是词牌名或者正文
            if '（' in li1 and '）' in li1:
                line_words.append(li1[:li1.index("（")])
                line_words.append(li1[li1.index("）")+1:])
            else:
                line_words.append(li1)

    line_words.append(lines1[len(lines1)-1])

    print(len(line_words))

    fo = open('i'+filename, 'w',encoding="UTF-8")  
    for l in line_words:
        if len(l)>2:
            fo.write(l)  
            fo.write('\n')  
    fo.close() 

    return 'ok'

# def read_data(filename):
#     with open(filename, encoding="utf-8") as f:
#         data = f.read()
#     data = list(data)
#     return data

def read_data(filename):
    #   读取中文宋词
    line_words = []

    with open(filename,"r",encoding="UTF-8") as f:
        line = f.readline()
 
        while line:

            while '\n' in line:
                line = line.replace('\n','') # 换行符替换

            line_words.append(line)

            line=f.readline()

    # 将所有字符(汉字+标点符号)拆开
    words = []
    for lw in line_words:
        words += [w for w in lw]
        
    return words
############## def end ######## def end ######## def end #################

def build_dataset(words, n_words):
    unknow_char='UNK'
    # 取出现频率最高的词的数量组成字典(前n_words高频率字符)，不在字典中的字用unknow_char代替
    count = [[unknow_char, -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()

    for word, _ in count:
        dictionary[word] = len(dictionary)

    data = list()
    unk_count = 0

    for word in words:
        index = dictionary.get(word, 0)

        if index == 0:  # dictionary[unknow_char]
            unk_count += 1

        data.append(index)

    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

    # 将字典保存到当前路径,供后续使用
    f=open('dictionary.json', 'w')
    f.write(json.dumps(dictionary))
    f.close()

    f=open('reversed_dictionary.json', 'w')
    f.write(json.dumps(reversed_dictionary))
    f.close()

    return data, count, dictionary, reversed_dictionary
############### def end ######## def end ######## def end #################

def index_data(sentences, dictionary):
    unknow_char='UNK'
    # shape = sentences.shape # list没有shape属性
    shape = np.shape(sentences)
    sentences = np.reshape(sentences, [-1]) #sentences.reshape([-1])
    index = np.zeros_like(sentences, dtype=np.int32)
    for i in range(len(sentences)):
        try:
            index[i] = dictionary[sentences[i]]
        except KeyError:
            index[i] = dictionary[unknow_char]

    return index.reshape(shape)
############### def end ######## def end ######## def end #################

def get_train_data(vocabulary, dictionary, batch_size, num_steps):
    
    # 将vocabulary按照batch_size长度等分为长度为len_batch的数组
    len_batch = len(vocabulary) // batch_size

    # 训练文字的编号
    x_index = index_data(vocabulary, dictionary)

    # 标签的编号
    # 为训练文字的第二个字符
    # 最后一个文字的标签是没有的,给一个足够大的标签标号表示
    y_index = index_data(vocabulary[1:], dictionary)
    # y_index[-1]=len(dictionary)-1 #=
    y_index=np.append(y_index,len(dictionary))

    # 所有batch的训练和标签数组
    x_batches = np.zeros([batch_size, len_batch], dtype=np.int32)
    y_batches = np.zeros([batch_size, len_batch], dtype=np.int32)

    for i in range(batch_size):
        x_batches[i] = x_index[len_batch*i : len_batch*(i+1)]
        y_batches[i] = y_index[len_batch*i : len_batch*(i+1)]

    # 每次训练的文字长度为num_steps
    # 每个batch的训练次数为:
    epoch_size = len_batch // num_steps

    for i in range(epoch_size):
        x = x_batches[:, num_steps*i : num_steps*(i+1)]
        y = y_batches[:, num_steps*i : num_steps*(i+1)]
        yield(x, y)
