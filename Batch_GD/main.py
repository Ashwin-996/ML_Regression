import csv
import time
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

class BGD:
    def __init__(self, learning_rate, input_file, target_file):
        self.learning_rate = learning_rate
        self.input_file = input_file
        self.target_file = target_file
        self.params = [0]
        self.param_count = 1
        self.normalized_input_file = 'normalized.csv'
        self.mean = []
        self.stnd_dev = []
        self.m = 0.0

        self.x_3d = []
        self.y_3d = []
        self.z_3d = []

    def initialize_params(self):
        inp = open(self.input_file, 'r')
        
        csv_reader = csv.reader(inp)
        flag = False
        for line in csv_reader:
            if flag==False:
                for i in range(len(line)):
                    self.params.append(0)
                    self.param_count += 1
                flag = True
            self.m += 1

        inp.close()

    def get_mean_sd(self):
        inp = open(self.input_file, 'r')

        csv_reader = csv.reader(inp)
        for j in range(self.param_count-1):
            summ = 0.0
            counter = 0.0
            for line in csv_reader:
                summ += float(line[j])
                counter += 1
            self.mean.append(summ/counter)
            inp.seek(0)
            
            flag = 0.0
            for line in csv_reader:
                temp = float(line[j]) - self.mean[j]
                flag += temp*temp
            self.stnd_dev.append(flag/counter)
            inp.seek(0)


        inp.close()

    
    def normalize_input(self):
        inp = open(self.input_file, 'r')
        op = open(self.normalized_input_file, 'w')

        csv_reader = csv.reader(inp)
        csv_writer = csv.writer(op)
        for line in csv_reader:
            training_data = []
            for i in range(len(line)):
                training_data.append((float(line[i]) - self.mean[i])/self.stnd_dev[i])
            csv_writer.writerow(training_data)

        op.close()
        inp.close()
    

    def hypotheses(self, training_data):
        summ = 0
        for i in range(self.param_count):
            summ += self.params[i]*training_data[i]

        return summ


    def update_params(self):
        inp = open(self.normalized_input_file, 'r')
        op = open(self.target_file, 'r')

        inp_reader = csv.reader(inp)
        op_reader = csv.reader(op)

        stop = -9999999.9
        for j in range(self.param_count):
            slope = 0.0
            for line in inp_reader:
                training_data = [1]
                row = next(op_reader)
                target = float(row[0])
                for data in line:
                    training_data.append(float(data))
                slope += (target - self.hypotheses(training_data))*training_data[j]
            
            update = self.learning_rate*slope/self.m
            self.params[j] += update
            stop = max(stop, update)
            inp.seek(0)
            op.seek(0)

        #for mesh_plot
        summ = 0
        for line in inp_reader:
            training_data = [1]
            row = next(op_reader)
            target = float(row[0])
            for data in line:
                training_data.append(float(data))
            temp = target - self.hypotheses(training_data)
            summ += temp*temp

        error = summ/(2*self.m)
        print(error)
        time.sleep(0.2)
        self.z_3d.append(error)
        self.x_3d.append(self.params[0])
        self.y_3d.append(self.params[1])

        op.close()
        inp.close()

        return stop
    
    
    def plot_data(self):
        data_points = []
        tgt_points = []
        hyp_points = []


        inp = open(self.normalized_input_file, 'r')
        op = open(self.target_file, 'r')

        inp_reader = csv.reader(inp)
        
        for line in inp_reader:
            training_data = [1]
            data_points.append(float(line[0]))
            for data in line:
                training_data.append(float(data))
            hyp_points.append(self.hypotheses(training_data))

        inp.close()


        op_reader = csv.reader(op)
        
        for line in op_reader:
            tgt_points.append(float(line[0]))

        op.close()

        data_points = np.asarray(data_points)
        tgt_points = np.asarray(tgt_points)
        hyp_points = np.asarray(hyp_points)

        plt.scatter(data_points, tgt_points)
        plt.plot(data_points, hyp_points, label = 'hypotheses')
        plt.xlabel('Acidity of wine')
        plt.ylabel('Density of wine')
        plt.legend()
        plt.show()

    def plot_mesh(self):
        self.x_3d = np.asarray(self.x_3d)
        self.y_3d = np.asarray(self.y_3d)
        self.z_3d = np.asarray(self.z_3d)

        fig = go.Figure(data=[go.Mesh3d(x=self.x_3d, y=self.y_3d, z=self.z_3d)])
        fig.show()

    def plot_contour(self):
        X = np.asarray(self.x_3d)
        Y = np.asarray(self.y_3d)
        Z = np.asarray(self.z_3d)

        plt.tricontourf(X, Y, Z)
        plt.show()

a = BGD(0.1, 'linearX.csv', 'linearY.csv')
a.initialize_params()
a.get_mean_sd()
a.normalize_input()
stop_flag = 0.001/a.m
step_count = 0
while(True):
    stop = a.update_params()
    if stop < stop_flag:
        break
    step_count += 1

print(step_count)
print(a.params)
a.plot_data()
#a.plot_mesh()
#a.plot_contour()

