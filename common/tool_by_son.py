import sys, os
sys.path.append(os.pardir) 

class addpiont:
    def __init__(self , add , train_point):
        self.add = add
        self.train_point = train_point

    def run(self):
        if self.add not in self.train_point:
            self.train_point.append(self.add)
        elif self.add == 1:
            over_add = [num for num in self.train_point if 1<num<2]
            for i in over_add:
                self.train_point.remove(i)
            add_dx = 1 / (len(over_add) + 2) 
            self.add += add_dx
            while 1< self.add < 2-0.001:
                self.train_point.append(self.add)
                self.add += add_dx  
        elif self.add == 20:
            over_add = [num for num in self.train_point if 19<num<20]
            for i in over_add:
                self.train_point.remove(i)
            add_dx = 1 / (len(over_add) + 2) 
            self.add += add_dx - 1
            while 19< self.add < 20-0.001:
                self.train_point.append(self.add)
                self.add += add_dx
        else:
            over_add = [num for num in self.train_point if self.add - 1 <num< self.add + 1]
            for i in over_add:
                self.train_point.remove(i)
            self.train_point.append(self.add)
            print(len(over_add))
            if len(over_add) == 1:
                self.train_point.append(self.add - 0.5)
                self.train_point.append(self.add + 0.5)
            else:
                add_dx = 1 / ((len(over_add) - 1) / 2 + 2)

                add_minus = (self.add - 1 ) + add_dx
                while self.add - 1 < add_minus < self.add -0.001:
                    self.train_point.append(add_minus)
                    add_minus += add_dx

                add_plus = self.add + add_dx
                while self.add < add_plus < self.add + 1 -0.001:
                    self.train_point.append(add_plus)
                    add_plus += add_dx
        return self.train_point




