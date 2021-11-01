# -*- coding: utf-8 -*-
import sys,os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # __file__获取执行文件相对路径，整行为取上一级的上一级目录
sys.path.append(BASE_DIR)
import matplotlib.pyplot as plt
import numpy as np
x_test = np.load('20**4_x_test_L2_max_15.npy' , allow_pickle=True)
t_test_one = np.load('20**4_t_test_one.npy' , allow_pickle=True)
t_test_two = np.load('20**4_t_test_two.npy' , allow_pickle=True)
t_test_three = np.load('20**4_t_test_three.npy' , allow_pickle=True)
t_test_four = np.load('20**4_t_test_four.npy' , allow_pickle=True)
print(np.shape(x_test))


layer1 = [1,3,5,7,9,11,13,15,18]
layer2 = [1]

layer1_size = 15
fp_start =  layer1_size**0.5 * 100+225
fp_end   =  fp_start + 387+80

""
address = 2730

fig1 = plt.figure(1)
plt.plot(x_test[address])

plt.plot([fp_start,fp_start+1],[-1.5,1.5])
plt.plot([fp_end,fp_end+1],[-1.5,1.5])
plt.ylim(-1.5,1.5)
plt.savefig('%s'%address + "_in_for_four_FULL_max_15" , dpi=300)
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
        address = (one-1)*17*18 + two*18 #只有两层时
        print(t_test_one[address])
        print(t_test_two[address]) 
        print(t_test_three[address])
        
        #address = (one-1)*(5**3)*20 + two*(5**3) + three*(5**2) + four*(5) + five - 1
        #print(np.shape(x_test))

        fig1 = plt.figure(1)
        plt.plot(x_test[address])
        plt.ylim(-1.5,1.5)
        plt.savefig('%s'%name , dpi=200)
        plt.draw()#
        plt.pause(3)
        plt.close(fig1) 
"""