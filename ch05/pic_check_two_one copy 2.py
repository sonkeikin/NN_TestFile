# -*- coding: utf-8 -*-
import sys,os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # __file__获取执行文件相对路径，整行为取上一级的上一级目录
sys.path.append(BASE_DIR)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
x_test = np.load('20**2_x_test.npy' , allow_pickle=True)
print(np.shape(x_test))

address = 13

fig1 = plt.figure(1)
plt.plot(x_test[address])
plt.ylim(-1.5,1.5)
plt.savefig('%s'%address + "_layer1_namadeta" , dpi=200)
plt.draw()#
plt.pause(1)
plt.close(fig1)
"""
xxx = np.array(xxx)
data_a = pd.DataFrame(xxx)
writer = pd.ExcelWriter('test3.xlsx')
data_a.to_excel(writer,'page_3',float_format='%.5f')
writer.save()
"""