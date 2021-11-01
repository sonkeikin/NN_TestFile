# -*- coding: utf-8 -*-
import sys,os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # __file__获取执行文件相对路径，整行为取上一级的上一级目录
sys.path.append(BASE_DIR)
from common.optimizer import *
from tqdm import tqdm
import numpy as np
from common.simulator_for_train_two_one_1 import five_layer_simulator
from two_layer_net import TwoLayerNet

test_one_layer_sample = [1,2,3,4,5,6,7,8,9,10,11,12,12.5,13,13.5,14,15,16]
test_two_layer_sample = [1,2,3,4,5,6,7,8,9,10,11,12,12.5,13,13.5,14,15,16]
#1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16
#1,3,5,7,9,11,13,16
#1,4,8,12,16

#1,2,3,4,5,7,9,11,13,15,16

get_data = five_layer_simulator(currentTime_end = 2000,
                                Analytical_accuracy=10,
                                test_one_layer_sample = test_one_layer_sample,
                                test_two_layer_sample = test_two_layer_sample,)
x_test , t_test_one ,t_test_two  = get_data.run_simulator_test()
np.save('20**2_x_train.npy' , x_test)

#真值数据都是一样的，就不用在再保存了，直接用使FULL

""
np.save('20**2_t_train_one_.npy' , t_test_one)
np.save('20**2_t_train_two.npy' , t_test_two)
""
