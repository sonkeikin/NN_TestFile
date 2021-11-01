import numpy as np
import math
from tqdm import tqdm
import time 

def standardization(E):
    fature_max_point = max(E) if max(E)>abs(min(E)) else abs(min(E))#获取最大值
    if max(E) + abs(min(E)) > 0.01:
        for i in range(len(E)):
            E[i] = abs(E[i]) / fature_max_point #这里的abs()是为了把波和谷全都变成波 



class five_layer_simulator:
    def __init__(self, spacesize=320,layer_ait = 120, currentTime_end=1000, f=3e+9, excitationPoint = 1 ,  w=80
                 ,layer_thick=50 , Observation_point=2, Analytical_accuracy=1, con_layer = 2,
                 train_one_layer_sample=[1,5,10,15,20],
                 train_two_layer_sample=[1,5,10,15,20],
                 train_three_layer_sample=[1,5,10,15,20],
                 train_four_layer_sample=[1,5,10,15,20],
                 train_five_layer_sample=[1,5,10,15,20],
                 test_one_layer_sample=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
                 test_two_layer_sample=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], 
                 test_three_layer_sample=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
                 test_four_layer_sample=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
                 test_five_layer_sample=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]):
        self.spaceSize = spacesize
        self.currentTime_end = currentTime_end
        self.f = f
        self.w = w
        self.con_layer = con_layer#添加导体板的层数
        self.layer_thick = layer_thick
        self.layer_air = layer_ait
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
        fp_data_start = int((60-self.excitationPoint)+(60-self.Observation_point)+2*self.layer_thick  +10)
        fp_data_end = int((60-self.excitationPoint)+(60-self.Observation_point)+(2*self.layer_thick*(max(self.train_one_layer_sample)**0.5))+self.w)
        fp_data_len = fp_data_end - fp_data_start#这里要加上第一层的特征值的计算时间。
        print("需要截取范围" , fp_data_start , fp_data_end,"长度",fp_data_len)
        ###########计算相同介电常数的数字和个数
        same_er_layer_nums = [x for x in self.train_one_layer_sample if x in self.train_two_layer_sample]
        print("相同的介电常数为：",same_er_layer_nums,"个数为：",len(same_er_layer_nums))
        ##########################################################################
        print("------------------------------教師データを読み込む------------------------------")
        with tqdm(self.train_one_layer_sample) as t:
            for layer_1_relativeEpsilon in t:
                #计算电波在一层传播的耗时
                #layer1_time_cost = int(layer_1_relativeEpsilon ** 0.5 * 2 * self.layer_thick)
                for layer_2_relativeEpsilon in self.train_two_layer_sample:
                    for layer_3_relativeEpsilon in self.train_three_layer_sample:
                        for layer_4_relativeEpsilon in self.train_four_layer_sample:
                            for layer_5_relativeEpsilon in self.train_five_layer_sample:
                                if layer_1_relativeEpsilon != layer_2_relativeEpsilon:
                                    for c1 in range(self.spaceSize):
                                        if 0 <= c1 <= 120:
                                            self.relativeEpsilon[c1] = 1
                                        elif 121 <= c1 <= 121+ 1 *self.layer_thick:
                                            self.relativeEpsilon[c1] = layer_1_relativeEpsilon
                                        elif (121+1*self.layer_thick+1) <= c1 <= 121+ 2 *self.layer_thick:
                                            self.relativeEpsilon[c1] = layer_1_relativeEpsilon
                                        elif (121+2*self.layer_thick+1) <= c1 <= 121+ 3 *self.layer_thick:
                                            self.relativeEpsilon[c1] = layer_1_relativeEpsilon
                                        elif (121+3*self.layer_thick+1) <= c1 <= 121+ 4 *self.layer_thick:
                                            self.relativeEpsilon[c1] = layer_1_relativeEpsilon
                                        elif (121+4*self.layer_thick+1) <= c1 <= 121+ 5 *self.layer_thick:
                                            self.relativeEpsilon[c1] = layer_1_relativeEpsilon
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
                                        newE[170] = 0                                   
                                        newH[0:self.spaceSize - 1] = oldH[:self.spaceSize - 1] - self.CH[:self.spaceSize - 1] * (
                                                newE[1:self.spaceSize] - newE[:self.spaceSize - 1])
                                        oldE = newE.copy()
                                        oldH = newH.copy()
                                        Observation_point_E.append(oldE[self.Observation_point])
                                    ######################开始截取特征量####################                               
                                    #Observation_point_E = Observation_point_E[fp_data_start: fp_data_end]
                                    #######################特征量规格化######################
                                    standardization(Observation_point_E)
                                    #####################################################################
                                    x_train.append(Observation_point_E)
                                    t_train_one.append(layer_1_relativeEpsilon)
                                    t_train_two.append(layer_2_relativeEpsilon)
                                    t_train_three.append(layer_3_relativeEpsilon)
                                    t_train_four.append(layer_4_relativeEpsilon)
                                    t_train_five.append(layer_5_relativeEpsilon)
                                    #print("该层观测点电界为：" , Observation_point_E)
                                else:
                                    break
                                    #print("本次相等，跳过" ,layer_1_relativeEpsilon,layer_2_relativeEpsilon)
        x = np.array(x_train)
        x.reshape(-1,fp_data_len)
        t_train_one = np.array(t_train_one)
        t_train_two = np.array(t_train_two)
        t_train_three = np.array(t_train_three)
        t_train_four = np.array(t_train_four)
        t_train_five = np.array(t_train_five)
        t_train_one = t_train_one.reshape(-1,1)
        t_train_two = t_train_two.reshape(-1,1)
        t_train_three = t_train_three.reshape(-1,1)
        t_train_four = t_train_four.reshape(-1,1)
        t_train_five = t_train_five.reshape(-1,1)
        return x,t_train_one,t_train_two,t_train_three,t_train_four,t_train_five,fp_data_len

    def run_simulator_test(self):
        x_test = []
        t_test_one = []
        t_test_two = []
        t_test_three = []
        t_test_four = []
        t_test_five = []
        ##########################################################################
        time_cost_air = int((self.layer_air - self.excitationPoint)+(self.layer_air-self.Observation_point))
        ##########################################################################
        print("-----------------------------テストデータを読み込む-----------------------------") 
        with tqdm(self.test_one_layer_sample) as t:
            for layer_1_relativeEpsilon in t:
                time_cost_layer1 = int(2 * self.layer_thick * ((layer_1_relativeEpsilon)**0.5))
                for layer_2_relativeEpsilon in self.test_two_layer_sample:
                    time_cost_layer2 = int(2 * self.layer_thick * ((layer_2_relativeEpsilon)**0.5))
                    step = 0
                    max_point = 0
                    while step <= 1:
                    ###生成关于第一层的观测波
                        for c1 in range(self.spaceSize):
                            if 0 <= c1 <= self.layer_air:
                                self.relativeEpsilon[c1] = 1
                            elif self.layer_air+1 <= c1 <= self.layer_air + (1*self.layer_thick):
                                self.relativeEpsilon[c1] = layer_1_relativeEpsilon
                            elif self.layer_air+1 + 1*self.layer_thick <= c1 <= self.spaceSize:
                                self.relativeEpsilon[c1] = layer_2_relativeEpsilon

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
                            newE[1:self.spaceSize - 1] = self.CE1[1:self.spaceSize - 1] * oldE[1:self.spaceSize - 1] - self.CE2[1:self.spaceSize - 1] * (newH[1:self.spaceSize - 1] - newH[0:self.spaceSize - 2])  # 電界を更新する
                            #if self.currentTime_now in range(self.w):
                            if currentTime_now <= self.w:
                                newE[self.excitationPoint] += (1 - math.cos(((2 * math.pi) / self.w) * currentTime_now)) ** 2
                            else:
                                newE[self.excitationPoint] = newE[self.excitationPoint]
                                excitationPoint_list.append(newE[self.excitationPoint])
                            newE[0] = oldE[1] + ((v1 * self.dt - self.dx) / (v1 * self.dt + self.dx)) * (newE[1] - oldE[0])
                            ####导体板的判断
                            if step == 0:
                                newE[self.layer_air + (self.con_layer-1) * self.layer_thick] = 0
                            elif step == 1:
                                newH[0:self.spaceSize - 1] = oldH[:self.spaceSize - 1] - self.CH[:self.spaceSize - 1] * (
                                    newE[1:self.spaceSize] - newE[:self.spaceSize - 1])
                            ####
                            newH[0:self.spaceSize - 1] = oldH[:self.spaceSize - 1] - self.CH[:self.spaceSize - 1] * (
                                    newE[1:self.spaceSize] - newE[:self.spaceSize - 1])
                            oldE = newE.copy()
                            oldH = newH.copy()
                            Observation_point_E.append(oldE[self.Observation_point])

                        Observation_point_E = Observation_point_E[(time_cost_air + time_cost_layer1) : (time_cost_air + time_cost_layer1 + self.w)]  
                        ###寻找最大值还是规格化判断
                        if step == 0:
                            max_point = max(map(abs,Observation_point_E))
                            #print(max_point)
                        elif step == 1:
                            for i in range(len(Observation_point_E)):
                                Observation_point_E[i]=Observation_point_E[i] / max_point
                        
                        step=step+1
                    
                    ####前层特征值的规格化
                    xxxx = layer_1_relativeEpsilon / max(self.test_one_layer_sample)
                    #xxxx = layer_1_relativeEpsilon
                    layer1_point = [xxxx]
                    layer_fix = layer1_point * 40
                    ####规则化第二层
                    #for i in range(len(Observation_point_E)):
                        #Observation_point_E[i] = abs(Observation_point_E[i])
                    
                    Observation_point_E = layer_fix + Observation_point_E
                    x_test.append(Observation_point_E)
                    t_test_one.append(layer_1_relativeEpsilon)
                    t_test_two.append(layer_2_relativeEpsilon)

        x = np.array(x_test)
        #print(np.shape(x))
        x.reshape(-1,1)
        t_test_one = np.array(t_test_one)
        t_test_two = np.array(t_test_two)
        t_test_one = t_test_one.reshape(-1 , 1)
        t_test_two = t_test_two.reshape(-1 , 1)

        return x,t_test_one,t_test_two
