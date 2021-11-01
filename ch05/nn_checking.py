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
predicit_layers = 2 #本次实验的层数
add_list =[] #增加过的数据列表
test_num = 1   #实验的起始点
test_num_max = 7  #实验的总次数
learning_rate = 0.00005   #学习精度
hidden_size = 1000
output_size = 1
max_loss_list = 0

test_one_layer_sample =   [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
test_two_layer_sample =   [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
test_three_layer_sample = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
test_four_layer_sample =  [30]
test_five_layer_sample =  [30]

x_test = np.load('20**5_x_test_one.npy' , allow_pickle=True)
t_test_one = np.load('20**5_t_test_one.npy')
t_test_two = np.load('20**5_t_test_two.npy')
t_test_three = np.load('20**5_t_test_three.npy')
t_test_four = np.load('20**5_t_test_four.npy')
t_test_five = np.load('20**5_t_test_five.npy')
print(np.shape(x_test))
##############################################################################################################################
t_test_list = {1:t_test_one , 2:t_test_two , 3:t_test_three , 4:t_test_four , 5:t_test_five}
t_test = t_test_list[predicit_layers]

print("実験を始めた")
import_len = 394
network = TwoLayerNet(input_size=import_len , hidden_size=hidden_size, output_size=output_size)
network.load_params("layer1_test.pkl")

""
tuicezhi = np.abs((network.predict(x_test)))
max_loss_list = tuicezhi- t_test
max_loss = np.max(abs(max_loss_list))

""
#rint("推测误差为",max_loss_list)
#print("真实数值为",t_test)
print("すべて誤差の平均値は：",sum(abs(max_loss_list))/len(x_test))
print("最大推定誤差は：",max_loss)

layer1_size = ((np.size(test_two_layer_sample)-1)   * np.size(test_three_layer_sample)* np.size(test_four_layer_sample)*(np.size(test_five_layer_sample)))
layer2_size = (np.size(test_three_layer_sample) * np.size(test_four_layer_sample) * ((np.size(test_five_layer_sample))))
layer3_size = (np.size(test_four_layer_sample)*((np.size(test_five_layer_sample))))
layer4_size = ((np.size(test_five_layer_sample)-1))
#####################要求第n层的位置，最大误差的索引 // layer_n_size   即可
yicengweizhi = test_one_layer_sample[np.argmax(max_loss_list) // layer1_size]
ercengweizhi = test_two_layer_sample[np.argmax(max_loss_list) // layer1_size // layer2_size]
sancengweizhi = test_two_layer_sample[np.argmax(max_loss_list) // layer1_size % layer2_size] 
if yicengweizhi >= ercengweizhi:
    ercengweizhi += 1
print("最大误差点位置" , yicengweizhi,ercengweizhi,sancengweizhi)
print("最大误差点的索引",np.argmax(max_loss_list))


"""#推测第二层的时候用的
yicengweizhi = test_one_layer_sample[np.argmax(max_loss_list) // layer1_size]
ercengweizhi = test_two_layer_sample[np.argmax(max_loss_list) // layer1_size // layer2_size]
sancengweizhi= test_three_layer_sample[np.argmax(max_loss_list) // layer1_size % layer2_size]
if sancengweizhi >= ercengweizhi:#如果第二层的位置出现在两层相同的地方之后的话，则实际的层数应该+1，因为相同那一层的数字被删掉了
    #比如二者相同， 都是5的话，则这是实际代表的位置是6
    sancengweizhi +=1

print("最大误差点位置" , yicengweizhi,ercengweizhi,sancengweizhi)
"""