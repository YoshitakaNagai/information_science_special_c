import random
import math
import numpy as np
import matplotlib.pyplot as plt
from operator import xor
from statistics import mean

#initial parameter
randn_param = 0.01
dimension = 2
alpha = 0.01
eps = 0.1



class Data:
    def __init__(self, size):
        self.X = np.empty((0, 2), float)
        self.T = np.empty((0, 1), float)
        self.size = size

class Parameter:
    def __init__(self, W, theta, size):
        self.W = W
        self.W_log = np.empty((0, 2), float)
        self.theta = theta
        self.theta_log = np.empty((0, 1), float)
        self.size = size


file_name = "./dataset.txt"
try:
    size = 0
    Input = Data(size)
    file = open(file_name, "r")
    for line in file:
        if line[0] == "N":
            continue
        else:
            data = line.split()
            X_tmp = np.array([[float(data[1]), float(data[2])]])
            T_tmp = np.array([[int(data[3])]])
            Input.X = np.append(Input.X, X_tmp, axis=0)
            Input.T = np.append(Input.T, T_tmp, axis=0)
            Input.size += 1
            print("Loading now")
except Exception as error:
    print(error)
finally:
    file.close()
    

def sigmoid_func(x):
    if x > 709:
        sigmoid = 1.0
    elif x < -709:
        sigmoid = 0
    else:
        sigmoid = 1/(1 + math.exp(-x))
    return sigmoid


def sigmoid_derivative(x):
    sigmoid_derivative = sigmoid_func(x) * (1 - sigmoid_func(x))
    return sigmoid_derivative


def calc_output(Ta, Tb, W, theta):
    sigma_WX = Ta * W[0] + Tb * W[1]
    s = sigma_WX - float(theta)
    #s = sigma_WX + float(theta)
    output = sigmoid_func(s)
    return output, s


def accuracy_checker(correct_list):
    correct_cnt = 0
    for i in range(len(correct_list)):
        if correct_list[i] == 1:
            correct_cnt += 1
    return correct_cnt / len(correct_list)
            


if __name__ == '__main__':
    #initialize
    W_ij_a_init = randn_param * np.random.randn(2)
    W_ij_b_init = randn_param * np.random.randn(2)
    W_jk_init = randn_param * np.random.randn(2)
    theta_ij_a_init = randn_param * np.random.randn(1)
    theta_ij_b_init = randn_param * np.random.randn(1)
    theta_jk_init = randn_param * np.random.randn(1)
    size_init = 1
    
    Param_ij = []
    Param_ij.append(Parameter(W_ij_a_init, theta_ij_a_init, size_init))
    Param_ij.append(Parameter(W_ij_b_init, theta_ij_b_init, size_init))
    #Param_ij[0].W_log = np.append(Param_ij[0].W_log, np.array([W_ij_a_init]), axis=0)
    #Param_ij[1].W_log = np.append(Param_ij[1].W_log, np.array([W_ij_b_init]), axis=0)
    #Param_ij[0].theta_log = np.append(Param_ij[0].theta_log, np.array([theta_ij_a_init]), axis=0)
    #Param_ij[1].theta_log = np.append(Param_ij[1].theta_log, np.array([theta_ij_b_init]), axis=0)

    Param_jk = Parameter(W_jk_init, theta_jk_init, size_init)
    #Param_jk.W_log = np.append(Param_jk.W_log, np.array([W_jk_init]), axis=0)
    #Param_jk.theta_log = np.append(Param_jk.theta_log, np.array([theta_jk_init]), axis=0)
    
    learned_flag = False
    roop0 = 0
    index = np.arange(Input.size)
    
    #correct_list_size = int(Input.size / 2)
    correct_list_size = int(Input.size)
    correct_list = [0 for i in range(correct_list_size)]
   
    E_fw_log = []
    accuracy_log = []
    E_avr_log = []
    step = 0
    epoch = 0
    while learned_flag == False:
        roop0 = int(roop0 % Input.size)

        #forward
        #idx_a_rdm = int(np.random.choice(index, 1, replace=False))
        #idx_b_rdm = int(np.random.choice(index, 1, replace=False))
        #idx_rdm = int(np.random.choice(index, 1, replace=False))
        idx_rdm = int(np.random.choice(index - 1, 1, replace=False))
        
        ya, s_ij_a= calc_output(Input.X[idx_rdm][0], Input.X[idx_rdm][1], Param_ij[0].W, Param_ij[0].theta)
        yb, s_ij_b= calc_output(Input.X[idx_rdm][0], Input.X[idx_rdm][1], Param_ij[1].W, Param_ij[1].theta)
        z, s_jk = calc_output(ya, yb, Param_jk.W, Param_jk.theta)
        print("z =", z)
        #delta_fw = z - xor(int(Input.T[idx_a_rdm]), int(Input.T[idx_b_rdm]))
        delta_fw = z - float(Input.T[idx_rdm])
        
        E_fw = delta_fw**2
        E_fw_log.append(E_fw)
        
        print("Eroor = ", E_fw)

        #learn W
        delta_W_jk_a = delta_fw * (1.0 - z) * z * ya
        delta_W_jk_b = delta_fw * (1.0 - z) * z * yb
        Param_jk.W[0] -= alpha * delta_W_jk_a
        Param_jk.W[1] -= alpha * delta_W_jk_b

        delta_W_ij_aa = delta_fw * (1.0 - z) * z * Param_jk.W[0] * (1.0 - ya) * ya * float(Input.X[idx_rdm][0])
        delta_W_ij_ab = delta_fw * (1.0 - z) * z * Param_jk.W[0] * (1.0 - ya) * ya * float(Input.X[idx_rdm][1])
        Param_ij[0].W[0] -= alpha * delta_W_ij_aa
        Param_ij[0].W[1] -= alpha * delta_W_ij_ab
        
        delta_W_ij_ba = delta_fw * (1.0 - z) * z * Param_jk.W[1] * (1.0 - yb) * yb * float(Input.X[idx_rdm][0])
        delta_W_ij_bb = delta_fw * (1.0 - z) * z * Param_jk.W[1] * (1.0 - yb) * yb * float(Input.X[idx_rdm][1])
        Param_ij[1].W[0] -= alpha * delta_W_ij_ba
        Param_ij[1].W[1] -= alpha * delta_W_ij_bb

        #learn theta
        delta_theta_jk = delta_fw * (1.0 - z) * z 
        Param_jk.theta -= alpha * delta_theta_jk

        delta_theta_ij_a = delta_theta_jk * Param_jk.theta * (1.0 - ya) * ya
        Param_ij[0].theta -= alpha * delta_theta_ij_a
        
        delta_theta_ij_b = delta_theta_jk * Param_jk.theta * (1.0 - yb) * yb
        Param_ij[1].theta -= alpha * delta_theta_ij_b

        if E_fw < eps:
            correct_list[roop0] = 1
        else:
            correct_list[roop0] = 0



        if roop0 == Input.size:
            epoch += 1
        
        if step % int(Input.size) == 0 or epoch == 1000:
            print("epoch : ", epoch)
            accuracy = accuracy_checker(correct_list)
            accuracy_log.append(accuracy)
            print("accuracy : ", accuracy)
            
            print("idx_rdm : ", idx_rdm)
            
            x1 = np.arange(-10.0, 10.0, 0.1)
            x2_a = (Param_ij[0].theta - Param_ij[0].W[0] * x1) / Param_ij[0].W[1]
            x2_b = (Param_ij[1].theta - Param_ij[1].W[0] * x1) / Param_ij[1].W[1]
            
            plt.clf()
            #plt.cla()
            plt.subplot(3,1,1)
            plot_line = plt.plot(x1, x2_a, label="a")
            plot_line = plt.plot(x1, x2_b, label="b")
            """
            for i in range(Input.size):
                plt_dataset = plt.scatter(Input.X[i,0], Input.X[i,1], c="green")
            """
            plt.title("line")
            plt.xlabel("x1")
            plt.ylabel("x2")
            plt.xlim(-1.0, 2.0)
            plt.ylim(-1.0, 2.0)
            
            plt.subplots_adjust(wspace=0.4, hspace=0.6)
            
            #plt.figure(figsize=(0, len(accuracy_log)))
            plt.subplot(3,1,2)
            plot_E = plt.plot(np.arange(len(accuracy_log)), accuracy_log, label="accuracy")
            plt.title("accuracy")
            plt.xlabel("epoch")
            plt.ylabel("rate")
            

            plt.subplots_adjust(wspace=0.4, hspace=0.6)
            
            E_avr = mean(E_fw_log)
            E_avr_log.append(E_avr)
            plt.subplot(3,1,3)
            plot_E = plt.plot(np.arange(len(E_avr_log)), E_avr_log, label="Error")
            plt.title("error")
            plt.xlabel("epoch")
            plt.ylabel("error")
            
            E_fw_log.clear()

            #plt.show()       
            plt.pause(0.00000000000000000001)
           
            index = np.arange(Input.size)
        
        if epoch == 10000:
            break 

        if accuracy > 0.7:
            x1 = np.arange(-10.0, 10.0, 0.1)
            x2_a = (Param_ij[0].theta - Param_ij[0].W[0] * x1) / Param_ij[0].W[1]
            x2_b = (Param_ij[1].theta - Param_ij[1].W[0] * x1) / Param_ij[1].W[1]
            plot_line = plt.plot(x1, x2_a, label="a")
            plot_line = plt.plot(x1, x2_b, label="b")
            """
            for i in range(Input.size):
                if Input.T[i] == 0:
                    plt_dataset = plt.scatter(Input.X[i,0], Input.X[i,1], c="green")
                if Input.T[i] == 1:
                    plt_dataset = plt.scatter(Input.X[i,0], Input.X[i,1], c="blue")
            """
            plt.title("line")
            plt.xlabel("x1")
            plt.ylabel("x2")
            plt.xlim(-1.5, 2.0)
            plt.ylim(-1.5, 2.0)
            plt.show()

            #learned_flag = True
        
        roop0 += 1
        step += 1


    
