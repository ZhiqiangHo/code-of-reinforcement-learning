#!/usr/bin/env python
# encoding: utf-8
'''
@author: Zhiqiang Ho
@contact: 18279406017@163.com
@file: 01_actions.py
@time: 8/1/20 10:36 AM
@desc:
'''
import numpy as np
import ptan
q_vals = np.array([[1, 2, 3], [1, -1, 0]])
print(q_vals)

selector = ptan.actions.ArgmaxActionSelector()

print(selector(q_vals))