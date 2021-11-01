import numpy as np
import math
from tqdm import tqdm
import time 

class five_layer_simulator:
    def __init__(self, spacesize=310, currentTime_end=1000, f=3e+9, excitationPoint = 1 ,  w=120
                 ,layer_thick=50 , Observation_point=2, Analytical_accuracy=10, train_one_layer_sample=[1,5,10,15,20],
                 train_two_layer_sample=[1,5,10,15,20],train_three_layer_sample=[1,5,10,15,20],train_four_layer_sample=[1,5,10,15,20],train_five_layer_sample=[1,5,10,15,20],test_one_layer_sample=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
                 test_two_layer_sample=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20] , test_three_layer_sample=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],test_four_layer_sample=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],test_five_layer_sample=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]):
        self.spaceSize = spacesize
        self.currentTime_end = currentTime_end
        self.f = f
        self.w = w
        self.layer_thick = layer_thick
        self.excitationPoint = excitationPoint
        self.Observation_point = Observation_point
        self.Analytical_accuracy = Analytical_accuracy
        self.train_one_layer_sample = train_one_layer_sample
        self.train_two_layer_sample = train_two_layer_sample
        self.train_three_layer_sample = train_three_layer_sample
        self.train_four_layer_sample = train_four_layer_sample
        self.train_five_layer_sample = train_five_layer_sample
        self.test_one_layer_sample = test_one_layer_sample
        self.test_two_layer_sample = test_two_layer_sample
        self.test_three_layer_sample = test_three_layer_sample
        self.test_four_layer_sample = test_four_layer_sample
        self.test_five_layer_sample = test_five_layer_sample
        self.c = 3.00e+8
        self.wavelength = self.c / self.f
        self.dx = self.wavelength / 10 * self.Analytical_accuracy
        self.dt = self.dx / self.c
        self.mu0 = 4 * math.pi * 1e-7
        self.epsilon0 = 1 / (self.mu0 * (self.c ** 2))
        self.relativeEpsilon = np.zeros(self.spaceSize)  # 媒質の中に、空く各点の誘電率を作る
        self.relativeMu = np.zeros(self.spaceSize - 1)  # 媒質の中に、空く各点の透磁率を作る
        self.relativeMu = np.ones_like(self.relativeMu)  # 媒質の中に、各点の透磁率の変化を表す
        self.Sigma = np.zeros(self.spaceSize)  # 媒質の中に、空く各点の導電率を作る
        self.CE1 = np.zeros(self.spaceSize)
        self.CE2 = np.zeros(self.spaceSize)
        self.CH = np.zeros(self.spaceSize - 1)
    
    def run_simulator_train(self):
        x_train = []
        t_train_one = []
        t_train_two = []
        t_train_three = []
        t_train_four = []
        t_train_five = []
        ##########################################################################
        fast_fp_data_start = 58+59
        fast_fp_data_end = fast_fp_data_start+self.w
        fp_data_start = int((60-self.excitationPoint)+(60-self.Observation_point)+2*self.layer_thick + 20)
        fp_data_end = int((60-self.excitationPoint)+(60-self.Observation_point)+(2*self.layer_thick*(20**0.5))+self.w)
        fp_data_len = fp_data_end - fp_data_start#这里要加上第一层的特征值的计算时间。
        print("需要截取范围" , fp_data_start , fp_data_end,"长度",fp_data_len)
        ##########################################################################
        print("------------------------------教師データを読み込む------------------------------")
        with tqdm(self.train_one_layer_sample) as t:
            for layer_1_relativeEpsilon in t:
                for layer_2_relativeEpsilon in self.train_two_layer_sample:
                    for layer_3_relativeEpsilon in self.train_three_layer_sample:
                        for layer_4_relativeEpsilon in self.train_four_layer_sample:
                            for layer_5_relativeEpsilon in self.train_five_layer_sample:
                                for c1 in range(self.spaceSize):
                                    if 0 <= c1 <= 60:
                                        self.relativeEpsilon[c1] = 1
                                    elif 61 <= c1 <= 60+1*self.layer_thick:
                                        self.relativeEpsilon[c1] = layer_1_relativeEpsilon
                                    elif 111 <= c1 <= 60+2*self.layer_thick:
                                        self.relativeEpsilon[c1] = layer_2_relativeEpsilon
                                    elif 161 <= c1 <= 60+3*self.layer_thick:
                                        self.relativeEpsilon[c1] = layer_2_relativeEpsilon
                                    elif 211 <= c1 <= 60+4*self.layer_thick:
                                        self.relativeEpsilon[c1] = layer_2_relativeEpsilon
                                    elif 261 <= c1 <= 60+5*self.layer_thick:
                                        self.relativeEpsilon[c1] = layer_2_relativeEpsilon
                                #print(self.relativeEpsilon)
                                for i in range(self.spaceSize):
                                    self.CE1[i] = (2 * self.epsilon0 * self.relativeEpsilon[i] - self.Sigma[i] * self.dt) / (
                                            2 * self.epsilon0 * self.relativeEpsilon[i] + self.Sigma[i] * self.dt)  # 初始のCE1を算出する
                                    self.CE2[i] = 2 * self.dt / (self.dx * (
                                            2 * self.epsilon0 * self.relativeEpsilon[i] + self.Sigma[i] * self.dt))  # 初始のCE2を算出する
                                for i in range(self.spaceSize - 1):
                                    self.CH[i] = self.dt / (self.dx * self.mu0 * self.relativeMu[i])
                                v1 = 1 / math.sqrt(
                                    ((self.mu0 * self.epsilon0 * (self.relativeEpsilon[0] + self.relativeEpsilon[1])) / 2))
                                #print("错误point" , ((( self.relativeEpsilon[self.spaceSize - 1]))))
                                v2 = 1 / math.sqrt(((self.mu0 * self.epsilon0 * (self.relativeEpsilon[self.spaceSize - 2] + self.relativeEpsilon[self.spaceSize - 1])) / 2))
                                newE = np.zeros(self.spaceSize)  # 空く更新した各点の電界の大きさを作る
                                oldE = np.zeros(self.spaceSize)  # 空く古い各点の電界の大きさを作る
                                newH = np.zeros(self.spaceSize - 1)  # 空く更新した各点の磁界の大きさを作る
                                oldH = np.zeros(self.spaceSize - 1)  # 空く古い各点の磁界の大きさを作る
                                Observation_point_E = []
                                excitationPoint_list = []
                                for currentTime_now in range(self.currentTime_end):
                                    #print(currentTime_now)
                                    newE[1:self.spaceSize - 1] = self.CE1[1:self.spaceSize - 1] * oldE[1:self.spaceSize - 1] - self.CE2[1:self.spaceSize - 1] * (newH[1:self.spaceSize - 1] - newH[0:self.spaceSize - 2])  # 電界を更新する
                                    #if self.currentTime_now in range(self.w):
                                    if currentTime_now <= self.w:
                                        newE[self.excitationPoint] += (1 - math.cos(((2 * math.pi) / self.w) * currentTime_now)) ** 2
                                    else:
                                        newE[self.excitationPoint] = newE[self.excitationPoint]
                                        excitationPoint_list.append(newE[self.excitationPoint])
                                    newE[0] = oldE[1] + ((v1 * self.dt - self.dx) / (v1 * self.dt + self.dx)) * (newE[1] - oldE[0])
                                    newE[self.spaceSize - 1] = oldE[self.spaceSize - 2] + (
                                            (v2 * self.dt - self.dx) / (v2 * self.dt + self.dx)) * (
                                                                        newE[self.spaceSize - 2] - oldE[self.spaceSize - 1])
                                    newH[0:self.spaceSize - 1] = oldH[:self.spaceSize - 1] - self.CH[:self.spaceSize - 1] * (
                                            newE[1:self.spaceSize] - newE[:self.spaceSize - 1])
                                    oldE = newE.copy()
                                    oldH = newH.copy()
                                    Observation_point_E.append(oldE[self.Observation_point])
                                ######################开始截取特征量#####################
                                #截取第一层的特征量 
                                Observation_point_E_one = Observation_point_E[fast_fp_data_start:fast_fp_data_end]
                                Observation_point_E_one_max_point = abs(min(Observation_point_E_one))
                                for i in range(len(Observation_point_E_one)):
                                    Observation_point_E_one[i] = Observation_point_E_one_max_point
                                #截取第二层的特征量
                                Observation_point_E_two = Observation_point_E[fp_data_start:fp_data_end]
                                #######################特征量规格化######################
                                fature_max_index = int(Observation_point_E_two.index(max(Observation_point_E_two) if max(Observation_point_E_two)>abs(min(Observation_point_E_two)) else min(Observation_point_E_two)))#获取最大值的索引
                                if max(Observation_point_E_two) + abs(min(Observation_point_E_two)) > 0.01:
                                    if fature_max_index<60:
                                        for i in range(0,119):
                                            Observation_point_E_two[i] = 1
                                    elif fature_max_index > (fp_data_len - 60):
                                        for i in range(fp_data_len-120,fp_data_len):
                                            Observation_point_E_two[i] = 1
                                    else:
                                        for i in range(fature_max_index-59,fature_max_index+59):
                                            Observation_point_E_two[i] = 1
                                #将两个特征量合并
                                Observation_point_E = Observation_point_E_one + Observation_point_E_two
                                #####################################################################
                                x_train.append(Observation_point_E)
                                t_train_one.append(layer_1_relativeEpsilon)
                                t_train_two.append(layer_2_relativeEpsilon)
                                t_train_three.append(layer_3_relativeEpsilon)
                                t_train_four.append(layer_4_relativeEpsilon)
                                t_train_five.append(layer_5_relativeEpsilon)
                                #print("该层观测点电界为：" , Observation_point_E)
        x = np.array(x_train)
        x.reshape(len(self.train_one_layer_sample) * len(self.train_two_layer_sample) * len(self.train_three_layer_sample), -1)
        t_train_one = np.array(t_train_one)
        t_train_two = np.array(t_train_two)
        t_train_three = np.array(t_train_three)
        t_train_four = np.array(t_train_four)
        t_train_five = np.array(t_train_five)
        t_train_one = t_train_one.reshape(len(self.train_one_layer_sample) * len(self.train_two_layer_sample) * len(self.train_three_layer_sample) * len(self.train_four_layer_sample) * len(self.train_five_layer_sample) , 1)
        t_train_two = t_train_two.reshape(len(self.train_one_layer_sample) * len(self.train_two_layer_sample) * len(self.train_three_layer_sample)  * len(self.train_four_layer_sample) * len(self.train_five_layer_sample) , 1)
        t_train_three = t_train_three.reshape(len(self.train_one_layer_sample) * len(self.train_two_layer_sample) * len(self.train_three_layer_sample)  * len(self.train_four_layer_sample) * len(self.train_five_layer_sample) , 1)
        t_train_four = t_train_four.reshape(len(self.train_one_layer_sample) * len(self.train_two_layer_sample) * len(self.train_three_layer_sample)  * len(self.train_four_layer_sample) * len(self.train_five_layer_sample) , 1)
        t_train_five= t_train_five.reshape(len(self.train_one_layer_sample) * len(self.train_two_layer_sample) * len(self.train_three_layer_sample)  * len(self.train_four_layer_sample) * len(self.train_five_layer_sample) , 1)
        fp_data_len += self.w
        return x,t_train_one,t_train_two,t_train_three,t_train_four,t_train_five,fp_data_len

    def run_simulator_test(self):
        x_test = []
        t_test_one = []
        t_test_two = []
        t_test_three = []
        t_test_four = []
        t_test_five = []
        ##########################################################################
        fast_fp_data_start = 58+59
        fast_fp_data_end = fast_fp_data_start+self.w
        fp_data_start = int((60-self.excitationPoint)+(60-self.Observation_point)+2*self.layer_thick + 20)
        fp_data_end = int((60-self.excitationPoint)+(60-self.Observation_point)+(2*self.layer_thick*(20**0.5))+self.w)
        fp_data_len = fp_data_end - fp_data_start#这里要加上第一层的特征值的计算时间。
        print("需要截取范围" , fp_data_start , fp_data_end,"长度",fp_data_len)
        ##########################################################################
        print("-----------------------------テストデータを読み込む-----------------------------") 
        with tqdm(self.test_one_layer_sample) as t:
            for layer_1_relativeEpsilon in t:
                for layer_2_relativeEpsilon in self.test_two_layer_sample:
                    for layer_3_relativeEpsilon in self.test_three_layer_sample:
                        for layer_4_relativeEpsilon in self.test_four_layer_sample:
                            for layer_5_relativeEpsilon in self.test_five_layer_sample:
                                for c1 in range(self.spaceSize):
                                    if 0 <= c1 <= 60:
                                        self.relativeEpsilon[c1] = 1
                                    elif 61 <= c1 <= 60+ 1 *self.layer_thick:
                                        self.relativeEpsilon[c1] = layer_1_relativeEpsilon
                                    elif 111 <= c1 <= 60+ 2 *self.layer_thick:
                                        self.relativeEpsilon[c1] = layer_2_relativeEpsilon
                                    elif 161 <= c1 <= 60+ 3 *self.layer_thick:
                                        self.relativeEpsilon[c1] = layer_2_relativeEpsilon
                                    elif 211 <= c1 <= 60+ 4 *self.layer_thick:
                                        self.relativeEpsilon[c1] = layer_2_relativeEpsilon
                                    elif 261 <= c1 <= 60+ 5 *self.layer_thick:
                                        self.relativeEpsilon[c1] = layer_2_relativeEpsilon
                                #print(self.relativeEpsilon)
                                for i in range(self.spaceSize):
                                    self.CE1[i] = (2 * self.epsilon0 * self.relativeEpsilon[i] - self.Sigma[i] * self.dt) / (
                                            2 * self.epsilon0 * self.relativeEpsilon[i] + self.Sigma[i] * self.dt)  # 初始のCE1を算出する
                                    self.CE2[i] = 2 * self.dt / (self.dx * (
                                            2 * self.epsilon0 * self.relativeEpsilon[i] + self.Sigma[i] * self.dt))  # 初始のCE2を算出する
                                for i in range(self.spaceSize - 1):
                                    self.CH[i] = self.dt / (self.dx * self.mu0 * self.relativeMu[i])
                                v1 = 1 / math.sqrt(
                                    ((self.mu0 * self.epsilon0 * (self.relativeEpsilon[0] + self.relativeEpsilon[1])) / 2))
                                #print("错误point" , ((( self.relativeEpsilon[self.spaceSize - 1]))))
                                v2 = 1 / math.sqrt(((self.mu0 * self.epsilon0 * (self.relativeEpsilon[self.spaceSize - 2] + self.relativeEpsilon[self.spaceSize - 1])) / 2))
                                newE = np.zeros(self.spaceSize)  # 空く更新した各点の電界の大きさを作る
                                oldE = np.zeros(self.spaceSize)  # 空く古い各点の電界の大きさを作る
                                newH = np.zeros(self.spaceSize - 1)  # 空く更新した各点の磁界の大きさを作る
                                oldH = np.zeros(self.spaceSize - 1)  # 空く古い各点の磁界の大きさを作る
                                Observation_point_E = []
                                excitationPoint_list = []
                                for currentTime_now in range(self.currentTime_end):
                                    #print(currentTime_now)
                                    newE[1:self.spaceSize - 1] = self.CE1[1:self.spaceSize - 1] * oldE[1:self.spaceSize - 1] - self.CE2[1:self.spaceSize - 1] * (newH[1:self.spaceSize - 1] - newH[0:self.spaceSize - 2])  # 電界を更新する
                                    #if self.currentTime_now in range(self.w):
                                    if currentTime_now <= self.w:
                                        newE[self.excitationPoint] += (1 - math.cos(((2 * math.pi) / self.w) * currentTime_now)) ** 2
                                    else:
                                        newE[self.excitationPoint] = newE[self.excitationPoint]
                                        excitationPoint_list.append(newE[self.excitationPoint])
                                    newE[0] = oldE[1] + ((v1 * self.dt - self.dx) / (v1 * self.dt + self.dx)) * (newE[1] - oldE[0])
                                    newE[self.spaceSize - 1] = oldE[self.spaceSize - 2] + (
                                            (v2 * self.dt - self.dx) / (v2 * self.dt + self.dx)) * (
                                                                        newE[self.spaceSize - 2] - oldE[self.spaceSize - 1])
                                    newH[0:self.spaceSize - 1] = oldH[:self.spaceSize - 1] - self.CH[:self.spaceSize - 1] * (
                                            newE[1:self.spaceSize] - newE[:self.spaceSize - 1])
                                    oldE = newE.copy()
                                    oldH = newH.copy()
                                    Observation_point_E.append(oldE[self.Observation_point])
                                ######################开始截取特征量#####################
                                #截取第一层的特征量 
                                Observation_point_E_one = Observation_point_E[fast_fp_data_start:fast_fp_data_end]
                                Observation_point_E_one_max_point = abs(min(Observation_point_E_one))
                                for i in range(len(Observation_point_E_one)):
                                    Observation_point_E_one[i] = Observation_point_E_one_max_point
                                
                                #截取第二层的特征量
                                Observation_point_E_two = Observation_point_E[fp_data_start:fp_data_end]
                                
                                #######################特征量规格化######################
                                fature_max_index = int(Observation_point_E_two.index(max(Observation_point_E_two) if max(Observation_point_E_two)>abs(min(Observation_point_E_two)) else min(Observation_point_E_two)))#获取最大值的索引
                                if max(Observation_point_E_two) + abs(min(Observation_point_E_two)) > 0.01:
                                    if fature_max_index<60:
                                        for i in range(0,119):
                                            Observation_point_E_two[i] = 1
                                    elif fature_max_index > (fp_data_len - 60):
                                        for i in range(fp_data_len-120,fp_data_len):
                                            Observation_point_E_two[i] = 1
                                    else:
                                        for i in range(fature_max_index-59,fature_max_index+59):
                                            Observation_point_E_two[i] = 1
                                
                                #将两个特征量合并

                                Observation_point_E = Observation_point_E_one + Observation_point_E_two

                                #####################################################################
                                x_test.append(Observation_point_E)
                                t_test_one.append(layer_1_relativeEpsilon)
                                t_test_two.append(layer_2_relativeEpsilon)
                                t_test_three.append(layer_3_relativeEpsilon)
                                t_test_four.append(layer_4_relativeEpsilon)
                                t_test_five.append(layer_5_relativeEpsilon)
                                #print("该层观测点电界为：" , Observation_point_E)
        x = np.array(x_test)
        x.reshape(len(self.test_one_layer_sample) * len(self.test_two_layer_sample) * len(self.test_three_layer_sample) ,-1)
        t_test_one = np.array(t_test_one)
        t_test_two = np.array(t_test_two)
        t_test_three = np.array(t_test_three)
        t_test_four = np.array(t_test_four)
        t_test_five = np.array(t_test_five)
        t_test_one = t_test_one.reshape(len(self.test_one_layer_sample) * len(self.test_two_layer_sample) * len(self.test_three_layer_sample) * len(self.test_four_layer_sample) * len(self.test_five_layer_sample) , 1)
        t_test_two = t_test_two.reshape(len(self.test_one_layer_sample) * len(self.test_two_layer_sample) * len(self.test_three_layer_sample) * len(self.test_four_layer_sample) * len(self.test_five_layer_sample), 1)
        t_test_three = t_test_three.reshape(len(self.test_one_layer_sample) * len(self.test_two_layer_sample) * len(self.test_three_layer_sample) * len(self.test_four_layer_sample) * len(self.test_five_layer_sample), 1)
        t_test_four = t_test_four.reshape(len(self.test_one_layer_sample) * len(self.test_two_layer_sample) * len(self.test_three_layer_sample) * len(self.test_four_layer_sample) * len(self.test_five_layer_sample), 1)
        fp_data_len += self.w
        return x,t_test_one,t_test_two,t_test_three,t_test_four,t_test_five,fp_data_len
