# -*- coding: utf-8 -*-
import sys,os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # __file__获取执行文件相对路径，整行为取上一级的上一级目录
sys.path.append(BASE_DIR)
import matplotlib.pyplot as plt
import numpy as np
import time as time 



x_test = np.load('aaaaaaaa.npy' , allow_pickle=True)



"""
xxxxx = []
for i in range(2):
    xxxxx.append(x_test)
    print(i)
np.save('aaaaaaaa',xxxxx)
"""