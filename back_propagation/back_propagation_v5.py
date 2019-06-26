import random
import math
import numpy as np
import matplotlib.pyplot as plt
from operator import xor
from statistics import mean

#initial parameter
randn_param = 1.0
dimension = 2
alpha = 0.2
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
    sigmoid_derivative = sigmoid_func(x) * (1.0 - sigmoid_func(x))
    return sigmoid_derivative


def calc_output(x1, x2, W, theta):
    sigma_WX = x1 * W[0] + x2 * W[1]
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
    finish_flag = False
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
    a = 0
    b = 1
    while finish_flag == False:
        roop0 = roop0 % Input.size

        #forward
        idx_rdm = int(np.random.choice(index - 1, 1, replace=False))
        
        ya, s_ij_a= calc_output(Input.X[idx_rdm][0], Input.X[idx_rdm][1], Param_ij[a].W, Param_ij[a].theta)
        yb, s_ij_b= calc_output(Input.X[idx_rdm][0], Input.X[idx_rdm][1], Param_ij[b].W, Param_ij[b].theta)
        z, s_jk = calc_output(ya, yb, Param_jk.W, Param_jk.theta)
        print("z =", z)
        delta_fw = z - float(Input.T[idx_rdm])
        
        E_fw = delta_fw**2
        E_fw_log.append(E_fw)
        
        print("Eroor = ", E_fw)
        
        #backward
        delta_jk = (Param_jk.W[a] + Param_jk.W[b]) * delta_fw

        dEjka_dWjka = delta_fw * sigmoid_derivative(s_jk) * ya
        dEjkb_dWjkb = delta_fw * sigmoid_derivative(s_jk) * yb

        dEjk_dthetajk = -delta_fw * sigmoid_derivative(s_jk)

        dEija0_dWij = delta_jk * sigmoid_derivative(s_ij_a) * float(Input.X[idx_rdm][0])
        dEija1_dWij = delta_jk * sigmoid_derivative(s_ij_a) * float(Input.X[idx_rdm][1])
        dEijb0_dWij = delta_jk * sigmoid_derivative(s_ij_b) * float(Input.X[idx_rdm][0])
        dEijb1_dWij = delta_jk * sigmoid_derivative(s_ij_b) * float(Input.X[idx_rdm][1])

        dEija_dthetaij = -delta_jk * sigmoid_derivative(s_ij_a)
        dEijb_dthetaij = -delta_jk * sigmoid_derivative(s_ij_b)


        #learn W
        Param_jk.W[a] -= alpha * dEjka_dWjka
        Param_jk.W[b] -= alpha * dEjkb_dWjkb

        Param_ij[a].W[0] -= alpha * dEija0_dWij
        Param_ij[a].W[1] -= alpha * dEija1_dWij
        Param_ij[b].W[0] -= alpha * dEijb0_dWij
        Param_ij[b].W[1] -= alpha * dEijb1_dWij

        #learn theta
        Param_jk.theta -= alpha * dEjk_dthetajk

        Param_ij[a].theta -= alpha * dEija_dthetaij
        Param_ij[b].theta -= alpha * dEijb_dthetaij

        if delta_fw < eps:
            correct_list[roop0] = 1
        else:
            correct_list[roop0] = 0


        # visualize        
        if step % Input.size == 0:
            epoch += 1
            """
            print("epoch : ", epoch)
            accuracy = accuracy_checker(correct_list)
            accuracy_log.append(accuracy)
            print("accuracy : ", accuracy)
            
            print("idx_rdm : ", idx_rdm)
            x1 = np.arange(-10.0, 10.0, 0.1)
            x2_a = (Param_ij[a].theta - Param_ij[a].W[0] * x1) / Param_ij[a].W[1]
            x2_b = (Param_ij[b].theta - Param_ij[b].W[0] * x1) / Param_ij[b].W[1]
            plt.clf()
            #plt.cla()
            plt.subplot(2,2,1)
            plot_line = plt.plot(x1, x2_a, label="a")
            plot_line = plt.plot(x1, x2_b, label="b")
            plt.title("line")
            plt.xlabel("x1")
            plt.ylabel("x2")
            plt.xlim(-1.0, 2.0)
            plt.ylim(-1.0, 2.0)
            
            plt.subplots_adjust(wspace=0.4, hspace=0.6)
            
            plt.subplot(2,2,3)
            plot_E = plt.plot(np.arange(len(accuracy_log)), accuracy_log, label="accuracy")
            plt.title("accuracy")
            plt.xlabel("epoch")
            plt.ylabel("rate")
            

            plt.subplots_adjust(wspace=0.4, hspace=0.6)
            
            E_avr = mean(E_fw_log)
            E_avr_log.append(E_avr)
            plt.subplot(2,2,4)
            plot_E = plt.plot(np.arange(len(E_avr_log)), E_avr_log, label="Error")
            plt.title("error")
            plt.xlabel("epoch")
            plt.ylabel("error")
        
            plt.pause(0.00000000000000000001)
            """

            accuracy = accuracy_checker(correct_list)
            accuracy_log.append(accuracy)
            
            E_avr = mean(E_fw_log)   
            E_avr_log.append(E_avr)

            if epoch == 1 or epoch % 50 == 0 or learned_flag == True:
                print("epoch : ", epoch)
                print("accuracy : ", accuracy)
                
                x1 = np.arange(-10.0, 10.0, 0.1)
                x2_a = (Param_ij[a].theta - Param_ij[a].W[0] * x1) / Param_ij[a].W[1]
                x2_b = (Param_ij[b].theta - Param_ij[b].W[0] * x1) / Param_ij[b].W[1]
                
                #plt.clf()
                #plt.cla()
                plt.subplot(2,2,1)
                plot_line = plt.plot(x1, x2_a, label="a")
                plot_line = plt.plot(x1, x2_b, label="b")
                
                for i in range(Input.size):
                    if Input.T[i] == 0:
                        plt_dataset = plt.scatter(Input.X[i,0], Input.X[i,1], c="green", s=5)
                    else:
                        plt_dataset = plt.scatter(Input.X[i,0], Input.X[i,1], c="red", s=5)


                plt.title("line")
                plt.xlabel("x1")
                plt.ylabel("x2")
                plt.xlim(-1.0, 2.0)
                plt.ylim(-1.0, 2.0)
                
                plt.subplots_adjust(wspace=0.4, hspace=0.6)
                
                plt.subplot(2,2,3)
                plot_E = plt.plot(np.arange(len(accuracy_log)), accuracy_log, label="accuracy")
                plt.title("accuracy")
                plt.xlabel("epoch")
                plt.ylabel("rate")
                

                plt.subplots_adjust(wspace=0.4, hspace=0.6)
                
                plt.subplot(2,2,4)
                plot_E = plt.plot(np.arange(len(E_avr_log)), E_avr_log, label="Error")
                plt.title("error")
                plt.xlabel("epoch")
                plt.ylabel("error")
                
                #plt.pause(10.0)
                plt.show()
            
            E_fw_log.clear()

           
            index = np.arange(Input.size)
        
            if accuracy == 1:
                print("complete learning!!")
                learned_flag = True


        if epoch == 1000:
            print("fail to learn")
            break 
                
        roop0 += 1
        step += 1


    
