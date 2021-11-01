# -*- coding: utf-8 -*-
import sys,os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # __file__获取执行文件相对路径，整行为取上一级的上一级目录
sys.path.append(BASE_DIR)
from common.optimizer import *
from tqdm import tqdm
import numpy as np
from common.simulator_for_train_four_one import five_layer_simulator
from two_layer_net import TwoLayerNet

test_one_layer_sample =   [1,3,5,7,9,11,13,15]
test_two_layer_sample =   [1,3,5,7,9,11,13,15]
test_three_layer_sample = [1,3,5,7,9,11,13,15]
test_four_layer_sample =  [1,3,5,7,9,11,13,15]
test_five_layer_sample =  [15]


get_data = five_layer_simulator(currentTime_end = 2000,
                                Analytical_accuracy=10,
                                test_one_layer_sample = test_one_layer_sample,
                                test_two_layer_sample = test_two_layer_sample,
                                test_three_layer_sample = test_three_layer_sample,
                                test_four_layer_sample = test_four_layer_sample,
                                test_five_layer_sample = test_five_layer_sample)
x_test , t_test_one ,t_test_two , t_test_three , t_test_four , t_test_five , date_len  = get_data.run_simulator_test()
print(np.shape(x_test))
np.save('20**4_x_test_L1_max_15.npy' , x_test)

#真值数据都是一样的，就不用在再保存了，直接用使FULL
""
np.save('20**4_t_test_one.npy' , t_test_one)
np.save('20**4_t_test_two.npy' , t_test_two)
np.save('20**4_t_test_three.npy' , t_test_three)
np.save('20**4_t_test_four.npy' , t_test_four)
np.save('20**4_t_test_five.npy' , t_test_five)
""