#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from utils import fix_data

from flags import parse_args
FLAGS, unparsed = parse_args()

# 处理语料库
_ = fix_data(FLAGS.Origin)
