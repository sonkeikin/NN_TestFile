# -*- coding: utf-8 -*-
import sys,os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # __file__获取执行文件相对路径，整行为取上一级的上一级目录
sys.path.append(BASE_DIR)
import matplotlib.pyplot as plt
import numpy as np
x_test = np.load('20**3_x_test_L1_max_15.npy' , allow_pickle=True)
t_test_one = np.load('20**3_t_test_one.npy' , allow_pickle=True)
t_test_two = np.load('20**3_t_test_two.npy' , allow_pickle=True)
t_test_three = np.load('20**3_t_test_three.npy' , allow_pickle=True)
t_test_four = np.load('20**3_t_test_four.npy' , allow_pickle=True)
print(np.shape(x_test))


layer1 = [7]
layer2 = [1,4,8,12,15]
layer3 = [1,4,8,12,15]

print(np.shape(x_test))
""
address = 31


""
fig1 = plt.figure(1)
plt.plot(x_test[address])
plt.ylim(-1.5,1.5)
plt.savefig('%s'%address + "_in_for_three_F1_max_15" , dpi=300)
plt.draw()#
plt.pause(1)
plt.close(fig1)

print("一层的介电常数为：",t_test_one[address])
print("二层的介电常数为：",t_test_two[address])
print("三层的介电常数为：",t_test_three[address])
print("四层的介电常数为：",t_test_four[address])
"""
for i in layer1:
    for j in layer2:
        one = i
        two = j
        print(len(x_test),one,two)
        name = 'fature_full:' + str(one) + '@' + str(two)
        #three = 1
        #four = 1
        #five = 1 
        if one <= two:
            two -= 1
        two -= 1
        #address = (one-1)*7*8 + two*8#只有两层时

        
        address = (one-1)*15*14 + two*15 
        #print(np.shape(x_test))
        print(t_test_one[address])
        print(t_test_two[address]) 
        print(t_test_three[address])
        fig1 = plt.figure(1)

        plt.plot(x_test[address])
        plt.Text(300,1.3,'L1 = %s'%one,'L2 = %s'%two)
        plt.ylim(-1.5,1.5)
        plt.savefig('%s'%name , dpi=200)
        plt.draw()#
        plt.pause(3)
        plt.close(fig1) 
        print("一层的介电常数为：",t_test_one[address])
        print("二层的介电常数为：",t_test_two[address])
        print("三层的介电常数为：",t_test_three[address])

        #print("四层的介电常数为：",t_test_four[address])
"""