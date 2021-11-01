# -*- coding: utf-8 -*-
import sys,os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # __file__获取执行文件相对路径，整行为取上一级的上一级目录
sys.path.append(BASE_DIR)
import numpy as np



t_text = np.load('1_to_20_text_five.npy')#载入1-10的数据
t_text2 = np.load('1_to_20_text_five.npy')#载入11-20的数据
t_text = np.append(t_text , t_text2 , axis=0)#合并数据
np.save('20*5_t_test_five.npy' , t_text)#保存合并后的数据