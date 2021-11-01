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
#from common.simulator_for_train_three_one import five_layer_simulator
from two_layer_net import TwoLayerNet
import tools as tool



time_start = time.time()
time_start_learnt = 0
add = 0 #最初的追加数据
add_time = 1 #追加数据的次数
predicit_layers = 1 #本次实验的层数##################训练对应的层数###################
add_list =[] #增加过的数据列表
test_num = 1   #实验的起始点
test_num_max = 30  #实验的总次数
learning_rate = 0.001   #学习精度
input_size = 1000
hidden_size = 500
output_size = 1
fp_data_len = 120

test_num_max_list = []
test_num_time_list = []
test_num_loss = []
test_num_max_rule = 20

train_one_layer_sample  =  [1,4,8,12,15,16]
train_two_layer_sample  =  [1,4,8,12,15,16]

x_test = np.load('20**2_x_test.npy' , allow_pickle=True)
#t_test_one = np.load('20**2_t_test_one.npy')；破
t_test = np.load('20**2_t_test_two.npy')

x_train = np.load('20**2_x_train.npy')
#t_train = np.load('20**2_t_train_one.npy')
t_train = np.load('20**2_t_train_two.npy')

add_1 = 0
add_2 = 0

test_point_num_list_1 = []
test_point_num_list_2 = []


##############################################################################################################################
"""
get_data = five_layer_simulator(currentTime_end = input_size,
                                   train_one_layer_sample=train_one_layer_sample,
                                   train_two_layer_sample=train_two_layer_sample,
                                   train_three_layer_sample=train_three_layer_sample,
                                   train_four_layer_sample = train_four_layer_sample,
                                   train_five_layer_sample = train_five_layer_sample)
x_train , t_train_one ,t_train_two , t_train_three , t_train_four ,t_train_five , fp_data_len= get_data.run_simulator_train()
print("教师数据的长度是：",fp_data_len)
"""
##############################################################################################################################
while test_num_max_rule > 1:
    time_start_learnt = time.time()
    test_num_loss = []
    train_one_layer_sample.sort()###########
    train_two_layer_sample.sort()###########

    print("============================================================ 第" , add_time , "回　教師データを追加実験 ==============================================================")
    """
    if add_1 != 0:
        get_data = five_layer_simulator(currentTime_end = input_size,
                                   train_one_layer_sample=[add_1],
                                   train_two_layer_sample=train_two_layer_sample,
                                   train_three_layer_sample=train_three_layer_sample,
                                   train_four_layer_sample = train_four_layer_sample)
        x_train_add , t_train_one_add ,t_train_two_add , t_train_three_add , t_train_four_add = get_data.run_simulator_train()
        x_train = np.append(x_train , x_train_add , axis=0)
        t_train_one = np.append(t_train_one , t_train_one_add , axis=0)
        t_train_two = np.append(t_train_two , t_train_two_add , axis=0)
        t_train_three = np.append(t_train_three , t_train_three_add , axis=0)
        t_train_four = np.append(t_train_four , t_train_four_add , axis=0)
    if add_2 != 0,:
        get_data = five_layer_simulator(currentTime_end = input_size,
                                   train_one_layer_sample=train_one_layer_sample,
                                   train_two_layer_sample=[add_2],
                                   train_three_layer_sample=train_three_layer_sample,
                                   train_four_layer_sample = train_four_layer_sample)
        x_train_add , t_train_one_add ,t_train_two_add , t_train_three_add , t_train_four_add = get_data.run_simulator_train()
        x_train = np.1.2,1.5,pend(x_train , x_train_add , axis=0)
        t_train_one = np.append(t_train_one , t_train_one_add , axis=0)
        t_train_two = np.append(t_train_two , t_train_two_add , axis=0)
        t_train_three = np.append(t_train_three , t_train_three_add , axis=0)
        t_train_four = np.append(t_train_four , t_train_four_add , axis=0)
    if add_3 != 0:
        get_data = five_layer_simulator(currentTime_end = input_size,
                                   train_one_layer_sample=train_one_layer_sample,
                                   train_two_layer_sample=train_two_layer_sample,
                                   train_three_layer_sample=[add_3],
                                   train_four_layer_sample = train_four_layer_sample)
        x_train_add , t_train_one_add ,t_train_two_add , t_train_three_add , t_train_four_add = get_data.run_simulator_train()
        x_train = np.append(x_train , x_train_add , axis=0)
        t_train_one = np.append(t_train_one , t_train_one_add , axis=0)
        t_train_two = np.append(t_train_two , t_train_two_add , axis=0)
        t_train_three = np.append(t_train_three , t_train_three_add , axis=0)
        t_train_four = np.append(t_train_four , t_train_four_add , axis=0)
    if add_4 != 0:
        get_data = five_layer_simulator(currentTime_end = input_size,
                                   train_one_layer_sample=train_one_layer_sample,
                                   train_two_layer_sample=train_two_layer_sample,
                                   train_three_layer_sample=train_three_layer_sample,
                                   train_four_layer_sample = [add_4])
        x_train_add , t_train_one_add ,t_train_two_add , t_train_three_add , t_train_four_add = get_data.run_simulator_train()
        x_train = np.append(x_train , x_train_add , axis=0)
        t_train_one = np.append(t_train_one , t_train_one_add , axis=0)
        t_train_two = np.append(t_train_two , t_train_two_add , axis=0)
        t_train_three = np.append(t_train_three , t_train_three_add , axis=0)
        t_train_four = np.append(t_train_four , t_train_four_add , axis=0)
    """
    print("今回の教師データは " , train_one_layer_sample)
    print("今回の教師データは " , train_two_layer_sample)

    batch_size = int(len(x_train) / 1)

    # データの読み込み
    #t_train_list = {1:t_train_one , 2:t_train_two , 3:t_train_three , 4:t_train_four , 5:t_train_five}
    #t_test_list = {1:t_test_one , 2:t_test_two , 3:t_test_three , 4:t_test_four , 5:t_test_five}
    #t_train = t_train_list[predicit_layers]
    #t_test = t_test_list[predicit_layers]

    print("今回の教師データ量は" , np.shape(x_train))

    train_size = x_train.shape[0]
    test_max_loss_list_value = []
    test_max_loss_list_one = []
    test_max_loss_list_two = []
    test_max_loss_list_three = []
    test_max_loss_list_four = []
    test_max_loss_list_five = []

    print("実験を始めた")
    test_num = 1
    while test_num <= test_num_max:
        level1_test_time = 1
        network = TwoLayerNet(input_size=fp_data_len, hidden_size=hidden_size, output_size=output_size)
        train_ave_loss_list = []
        print("第",test_num,"回実験")
        ave_1 = max(t_test)
        ave_wave_1 = 1
        ave_wave_2 = 1
        learnt_num = 1
        aaaa = 0
        #while aaaa==0:      #进行学习 
        while ave_wave_1>0.0001 or ave_wave_2>0.0001:      #进行学习 
            aaaa +=1
            #time.sleep(0.7)
            ave_wave_2 = ave_wave_1#更新上次的计算波动
            train_ave_loss_list = []
            train_max_loss_list = []
            test_ave_loss_list = []
            test_max_loss_list = []
            learning_times = 10 * 500
            for j in tqdm(range(learning_times) , ascii=True,desc="学習進度"):
                
                if j <= learning_times - 500:
                    batch_mask = np.random.choice(train_size, batch_size)
                    x_batch = x_train[batch_mask]
                    t_batch = t_train[batch_mask]

                    grad = network.gradient(x_batch, t_batch)
                    """
                    # 更新 by AdaGrad
                    lr = learning_rate
                    h = {}
                    for key, val in network.params.items():
                        h[key] = np.zeros_like(val)
                    for key in network.params.keys():
                        h[key] += grad[key] * grad[key]
                        network.params[key] -= lr * grad[key] / (np.sqrt(h[key]) + 1e-7)
                    """
                    for key in ('W1', 'b1', 'W2', 'b2'):
                        network.params[key] -= learning_rate * grad[key]
                else:
                    batch_mask = np.random.choice(train_size, batch_size)
                    x_batch = x_train[batch_mask]
                    t_batch = t_train[batch_mask]

                    grad = network.gradient(x_batch, t_batch)
                    """
                    # 更新 by AdaGrad
                    lr = learning_rate
                    h = {}
                    for key, val in network.params.items():
                        h[key] = np.zeros_like(val)
                    for key in network.params.keys():
                        h[key] += grad[key] * grad[key]
                        network.params[key] -= lr * grad[key] / (np.sqrt(h[key]) + 1e-7)
                    """
                    for key in ('W1', 'b1', 'W2', 'b2'):
                        network.params[key] -= learning_rate * grad[key]
                        
                    #train_ave_loss = network.accuracy(x_train, t_train)  # 这里的误差是所有的train数据的误差
                    #test_ave_loss = network.accuracy(x_test, t_test)  # 这里的误差是所有的test数据的总的平均误差
                    train_max_loss = np.max(np.abs(network.predict(x_train) - t_train))
                    test_max_loss = np.max(np.abs(network.predict(x_test) - t_test))

                    train_max_loss_list.append(train_max_loss)
                    test_max_loss_list.append(test_max_loss)
            learnt_num += 1
            ave_2 = np.average(test_max_loss_list)
            ave_wave_1 = abs(ave_1 - ave_2) / max(t_test)   
            print("第",level1_test_time , "回の結果：" , ave_2 , "结果波动：" , "%.2f%%" % (ave_wave_1 * 100))
            level1_test_time+=1
            ave_1 = ave_2#更新上次的计算结果
            #engine.say('level1 test has finished running') #载入朗读内容
            #engine.runAndWait() #启动发音程序
        ##########################保存参数##########################    
        network.save_params("two_two_max16.pkl")
    
        print("Saved Network Parameters!")
        #tool.sendmail(test_max_loss)
        ##########################################################
        max_loss_list = np.abs(network.predict(x_test) - t_test)
        max_loss = np.max(max_loss_list)
        max_loss_index = np.argmax(max_loss_list)
        yicengweizhi = t_test_one[max_loss_index] 
        ercengweizhi = t_test_two[max_loss_index]
        sancengweizhi = t_test_three[max_loss_index]
        sichengweizhi = t_test_four[max_loss_index]
        wuchengweizhi = t_test_five[max_loss_index]
        print("max loss index is :",max_loss_index)
        print("最大誤差が出た位置　" , yicengweizhi,ercengweizhi,sancengweizhi)
        test_max_loss_list_value.append(max_loss)
        test_max_loss_list_one.append(yicengweizhi)
        test_max_loss_list_two.append(ercengweizhi)
        test_max_loss_list_three.append(sancengweizhi)
        test_max_loss_list_four.append(sichengweizhi)
        test_max_loss_list_five.append(wuchengweizhi)
        test_num_loss.append(test_max_loss)
        print(test_num_loss)
        test_num_max_rule = sum(test_num_loss) / test_num 
        test_num += 1


###################计算出现次数做多的采样点################
    add_1 = tool.get_max_point(test_max_loss_list_one)
    #add_2 = tool.get_max_point(test_max_loss_list_two)#########
    #add_3 = tool.get_max_point(test_max_loss_list_three)#########
    #add_4 = tool.get_max_point(test_max_loss_list_four)#########

    print("一層目では追加したポイントは" , add_1)
    #print("二層目では追加したポイントは" , add_2)###########
    #print("三層目では追加したポイントは" , add_3)###########
    #print("四層目では追加したポイントは" , add_4)###########

    train_one_layer_sample = tool.point_append(add_1 , train_one_layer_sample)
    #train_two_layer_sample = tool.point_append(add_2 , train_two_layer_sample)#########
    #train_three_layer_sample = tool.point_append(add_3 , train_three_layer_sample)#########
    #train_four_layer_sample = tool.point_append(add_4 , train_four_layer_sample)#########

    now_test_need_time = time.time() - time_start_learnt
    print("今回の計算する時間" , datetime.timedelta(seconds=(time.time() - time_start_learnt))) 
    test_num_time_list.append(int(now_test_need_time))

    test_point_num_list_1.append(len(train_one_layer_sample))
    #test_point_num_list_2.append(len(train_two_layer_sample))###########
    #test_point_num_list_3.append(len(train_three_layer_sample))###########
    #test_point_num_list_4.append(len(train_four_layer_sample))###########

    print("今回の平均な最大テスト誤差" , test_num_max_rule) 
    test_num_max_list.append(test_num_max_rule)
    add_time += 1

###########################warning#########################
time_cost = datetime.timedelta(seconds=(time.time() - time_start))#计算程序耗时
print("総計学習時間:" , time_cost) #显示耗时，by hms format
#engine = pyttsx3.init()  #载入tts引擎
#engine.say('program has finished running om macmini') #载入朗读内容
#engine.runAndWait() #启动发音程序



########################################################################
tool.get_pic(test_num_max_list,time,label = "num_max_loss" , x_label = "max loss" , y_label = "time")
tool.get_pic(test_num_time_list,time,label="num_time" , x_label = "time(s)" , y_label = "time")
tool.get_pic(test_point_num_list_1,time,label="point_num_1" , x_label = "point num" , y_label = "time")
#tool.get_pic(test_point_num_list_2,time,label="point_num_2" , x_label = "point num" , y_label = "time")########
#tool.get_pic(test_point_num_list_3,time,label="point_num_3" , x_label = "point num" , y_label = "time")########
#tool.get_pic(test_point_num_list_4,time,label="point_num_4" , x_label = "point num" , y_label = "time")########

###########################显示增加的采样点分布情况#########################
print("最大誤差の変化" , test_num_max_list)
print("最適な一層目の教師データ" , train_one_layer_sample)
print("最適な二層目の教師データ" , train_two_layer_sample)
print("最適な三層目の教師データ" , train_three_layer_sample)
print("最適な四層目の教師データ" , train_four_layer_sample)
y = []
for i in range(len(train_two_layer_sample)):
    y.append(0)
plt.scatter(train_two_layer_sample,y,edgecolors='red')
plt.xlim(0,20)
plt.ylim(-1,1)
plt.show()
