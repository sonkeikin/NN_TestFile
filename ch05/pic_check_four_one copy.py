# -*- coding: utf-8 -*-
import sys,os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # __file__获取执行文件相对路径，整行为取上一级的上一级目录
sys.path.append(BASE_DIR)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
x_test = np.load('.npy' , allow_pickle=True)
print(np.shape(x_test))

xxx = []

for i in range(120):
    xxx.append(((x_test[0][i])))

for i in range(120):
    print(xxx[i])


address = 8


fig1 = plt.figure(1)
plt.plot(xxx)
plt.ylim(-4,4)
plt.savefig('%s'%address + "_in_for_four_F1_max_15" , dpi=300)
plt.draw()#
plt.pause(1)
plt.close(fig1)

xxx = np.array(xxx)
data_a = pd.DataFrame(xxx)
writer = pd.ExcelWriter('test3.xlsx')
data_a.to_excel(writer,'page_3',float_format='%.5f')
writer.save()
