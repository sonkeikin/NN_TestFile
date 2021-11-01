# -*- coding: utf-8 -*-
import sys,os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # __file__获取执行文件相对路径，整行为取上一级的上一级目录
sys.path.append(BASE_DIR)
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import datetime , random , time
from common.optimizer import *
from tqdm import tqdm
import numpy as np
from two_layer_net import TwoLayerNet
import tools as tool
import pandas as pd

time_start = time.time()
time_start_learnt = 0
add = 0 #最初的追加数据
add_time = 1 #追加数据的次数
predicit_layers = 1 #本次实验的层数
add_list =[] #增加过的数据列表
test_num = 1   #实验的起始点
test_num_max = 7  #实验的总次数
learning_rate = 0.0001   #学习精度
hidden_size = 500
output_size = 1
max_loss_list = 0

test_one_layer_sample =   [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
test_two_layer_sample =   [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
test_three_layer_sample = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
test_four_layer_sample =  [30]
test_five_layer_sample =  [30]

x_test = np.load('20**3_x_test.npy' , allow_pickle=True)
t_test_one = np.load('20**3_t_test_one.npy')
t_test_two = np.load('20**3_t_test_two.npy')
t_test_three = np.load('20**3_t_test_three.npy')
#t_test_four = np.load('20**3_t_test_four.npy')
#t_test_five = np.load('20**3_t_test_five.npy')
print(np.shape(x_test))
##############################################################################################################################
#t_test_list = {1:t_test_one , 2:t_test_two , 3:t_test_three , 4:t_test_four , 5:t_test_five}
#t_test = t_test_list[predicit_layers]
t_test = t_test_three
print("実験を始めた")
import_len = 120
network = TwoLayerNet(input_size=import_len , hidden_size=hidden_size, output_size=output_size)
network.load_params("two_two_max16.pkl")

""
tuicezhi = np.abs((network.predict(x_test)))
max_loss_list = tuicezhi- t_test
max_loss = np.max(max_loss_list)                #这里删除了一个abs（）
max_loss_index = np.argmax(max_loss_list)
""
#rint("推测误差为",max_loss_list)
#print("真实数值为",t_test)
print("すべて誤差の平均値は：",sum(abs(max_loss_list))/len(x_test))
print("============================")
print("最大推定誤差は：",max_loss)

max_loss = np.max(max_loss_list)
max_loss_index = np.argmax(max_loss_list)
yicengweizhi = t_test_one[max_loss_index] 
ercengweizhi = t_test_two[max_loss_index]
sancengweizhi = t_test_three[max_loss_index]
#sichengweizhi = t_test_four[max_loss_index]
#wuchengweizhi = t_test_five[max_loss_index]
print("推测值",tuicezhi[max_loss_index])
print("真值",t_test[max_loss_index])
print("位置" , "layer1 = ",yicengweizhi,"layer2 = ",ercengweizhi,"layer3 = ",sancengweizhi)
print("索引",max_loss_index)


"""#推测第二层的时候用的
yicengweizhi = test_one_layer_sample[np.argmax(max_loss_list) // layer1_size]
ercengweizhi = test_two_layer_sample[np.argmax(max_loss_list) // layer1_size // laysr2_size]
sancengweizhi= test_three_layer_sample[np.argmax(max_loss_list) // layer1_size % layer2_size]
if sancengweizhi >= ercengweizhi:#如果第二层的位置出现在两层相同的地方之后的话，则实际的层数应该+1，因为相同那一层的数字被删掉了
    #比如二者相同， 都是5的话，则这是实际代表的位置是6
    sancengweizhi +=1

print("最大误差点位置" , yicengweizhi,ercengweizhi,sancengweizhi)
"""