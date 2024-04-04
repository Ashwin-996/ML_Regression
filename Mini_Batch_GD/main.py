import csv
import numpy as np
from math import sqrt
import plotly.graph_objects as go

class SGD:
    def __init__(self, learning_rate, batch_size, input_file, k, stopping_crit):
        self.params = [3,1,2]
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.input_file = input_file
        self.param_count = 3
        self.m = 1000000
        self.k = k
        self.stopping_crit = stopping_crit

        self.x_3d = []
        self.y_3d = []
        self.z_3d = []

    def sample_data(self):
        file = open(self.input_file, 'w')
        csv_writer = csv.writer(file)
        for i in range(1000000):
            x1 = np.random.normal(3,2)
            x2 = np.random.normal(-1,2)
            noise = np.random.normal(0, sqrt(2))
            y = self.params[0] + self.params[1]*x1 + self.params[2]*x2 + noise

            line = [x1, x2, y]
            csv_writer.writerow(line)

        file.close()

    def hypotheses(self, training_data):
        summ = 0
        for i in range(self.param_count):
            summ += self.params[i]*training_data[i]

        return summ

    def update_params(self):
        inp = open(self.input_file, 'r')
        inp_reader = csv.reader(inp)

        total_iter = 0
        iter_num = 0
        avg_stop = 0
        prev_avg_stop = 0
        batch_num = 0
        while(True):
            data = []
            counter = 0
            for line in inp_reader:
                data.append(line)
                counter += 1
                if counter==self.batch_size:
                    break

            stop = -99999
            for j in range(self.param_count):
                slope = 0.0
                for line in data:
                    training_data = [1]
                    for i in range(len(line)-1):
                        training_data.append(float(line[i]))
                    target = float(line[-1])
                    slope += (target - self.hypotheses(training_data))*training_data[j]
                
                update = self.learning_rate*slope/self.batch_size
                self.params[j] += update
                stop = max(stop, update)

            #for 3d mesh plot
            self.x_3d.append(self.params[0])
            self.y_3d.append(self.params[1])
            self.z_3d.append(self.params[2])

            batch_num += 1
            iter_num += 1
            avg_stop += stop
            if iter_num==self.k:
                avg_stop = avg_stop/self.k
                print(abs(avg_stop - prev_avg_stop))
                if abs(avg_stop - prev_avg_stop) < self.stopping_crit:
                    return total_iter
                iter_num = 0
                prev_avg_stop = avg_stop
                avg_stop = 0

            if batch_num == self.m/self.batch_size:
                inp.seek(0)
                batch_num = 0
            
            total_iter += 1
            if total_iter > 5000:
                return total_iter

        inp.close()

    def mesh_plot(self):
        self.x_3d = np.asarray(self.x_3d)
        self.y_3d = np.asarray(self.y_3d)
        self.z_3d = np.asarray(self.z_3d)

        fig = go.Figure(data=[go.Mesh3d(x=self.x_3d, y=self.y_3d, z=self.z_3d)])
        fig.show()


a = SGD(0.001, 100, 'temp1.csv', 1000, 0.00000001)
a.sample_data()
a.params = [0,0,0]
total = a.update_params()
print(a.params)
print(total)
a.mesh_plot()


#r = 1, parameters = [2.9755622936110018, 1.0491919950249544, 1.9931626404303542]
#r = 100, parameters = [2.9978979329047584, 0.998662647399741, 2.0045660349702183]
#r = 10000,
      #if LR = 0.1 and SC=0.0001, parameters = [2.9861916268488637, 1.005014406846371, 1.9971958062748723] 
      #if LR = 0.001 an SC accordingly, parameters = [0.8576210204755945, 1.4529831055166549, 1.7997732755833473]
#r = 1000000, parameters = [0.030687902973343513, 0.12249698849900442, 0.033151600538047866] for LR = 0.001

#for q2test.csv,
      #if batch_size = 1, k = 100, parameters = [2.9835888045774843, 1.0131621819753607, 2.031992358642073]  
      #if batch_size = 100, k = 5, parameters = [3.0074063306021923, 1.0021290186783647, 2.0012311789550985]
      #if batch_size = 10000,
            #if LR = 0.01, parameters = [3.0068938143872304, 1.0005570393631404, 2.0014320667335457]
            #if LR = 0.001, parameters = [1.8711678248831392, 1.0136486736901118, 1.989473928540866]
