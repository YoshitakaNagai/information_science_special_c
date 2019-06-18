import random
import math
import numpy as np
import matplotlib.pyplot as plt
from operator import xor

#initial parameter
randn_param = 1
dimension = 2
alpha = 0.01
eps = 0.01



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
    Param_ij[0].W_log = np.append(Param_ij[0].W_log, np.array([W_ij_a_init]), axis=0)
    Param_ij[1].W_log = np.append(Param_ij[1].W_log, np.array([W_ij_b_init]), axis=0)
    Param_ij[0].theta_log = np.append(Param_ij[0].theta_log, np.array([theta_ij_a_init]), axis=0)
    Param_ij[1].theta_log = np.append(Param_ij[1].theta_log, np.array([theta_ij_b_init]), axis=0)

    Param_jk = Parameter(W_jk_init, theta_jk_init, size_init)
    Param_jk.W_log = np.append(Param_jk.W_log, np.array([W_jk_init]), axis=0)
    Param_jk.theta_log = np.append(Param_jk.theta_log, np.array([theta_jk_init]), axis=0)
    
    learned_flag = False
    roop0 = 0
    index = np.arange(Input.size)
    
    correct_list_size = int(Input.size / 2)
    correct_list = [0 for i in range(correct_list_size)]
    
    step = 0
    epoch = 0
    while learned_flag == False:
        roop0 = int(roop0 % (Input.size/2))

        #forward
        idx_a_rdm = int(np.random.choice(index, 1, replace=False))
        idx_b_rdm = int(np.random.choice(index, 1, replace=False))
        ya, s_ij_a= calc_output(Input.T[idx_a_rdm], Input.T[idx_b_rdm], Param_ij[0].W, Param_ij[0].theta)
        yb, s_ij_b= calc_output(Input.T[idx_a_rdm], Input.T[idx_b_rdm], Param_ij[1].W, Param_ij[1].theta)
        z, s_jk = calc_output(ya, yb, Param_jk.W, Param_jk.theta)
        print("z =", z)
        delta_fw = z - xor(int(Input.T[idx_a_rdm]), int(Input.T[idx_b_rdm]))
        E_fw = delta_fw**2
        print("Eroor = ", E_fw)
        #backward ... 誤差は(1 + 2)個?
        delta_bw_jk = (Param_jk.W[0] + Param_jk.W[1]) * delta_fw
        delta_bw_ij_a = Param_ij[0].W[0] * delta_bw_jk + Param_ij[0].W[1] * delta_fw
        delta_bw_ij_b = Param_ij[1].W[0] * delta_bw_jk + Param_ij[1].W[1] * delta_fw
        
        E_bw_jk = delta_bw_jk**2
        E_bw_ij_a = delta_bw_ij_a**2
        E_bw_ij_b = delta_bw_ij_b**2

        #learn W
        delta_W_jk_a = -2 *alpha * delta_bw_jk * sigmoid_derivative(s_jk) * ya
        delta_W_jk_b = -2 *alpha * delta_bw_jk * sigmoid_derivative(s_jk) * yb
        Param_jk.W[0] += delta_W_jk_a
        Param_jk.W[1] += delta_W_jk_b
        Param_jk.W_log = np.append(Param_jk.W_log, np.array([Param_jk.W]), axis=0)

        delta_W_ij_aa = -2 *alpha * delta_bw_ij_a * sigmoid_derivative(s_ij_a) * Input.X[idx_a_rdm][0]
        delta_W_ij_ab = -2 *alpha * delta_bw_ij_a * sigmoid_derivative(s_ij_a) * Input.X[idx_a_rdm][1]
        Param_ij[0].W[0] += delta_W_ij_aa
        Param_ij[0].W[1] += delta_W_ij_ab
        Param_ij[0].W_log = np.append(Param_ij[0].W_log, np.array([Param_ij[0].W]), axis=0)
        
        delta_W_ij_ba = -2 *alpha * delta_bw_ij_b * sigmoid_derivative(s_ij_b) * Input.X[idx_b_rdm][0]
        delta_W_ij_bb = -2 *alpha * delta_bw_ij_b * sigmoid_derivative(s_ij_b) * Input.X[idx_b_rdm][1]
        Param_ij[1].W[0] += delta_W_ij_ba
        Param_ij[1].W[1] += delta_W_ij_bb
        Param_ij[1].W_log = np.append(Param_ij[1].W_log, np.array([Param_ij[1].W]), axis=0)

        #learn theta
        delta_theta_jk = 2 * alpha * delta_bw_jk * sigmoid_derivative(s_jk)
        Param_jk.theta += delta_theta_jk
        Param_jk.theta_log = np.append(Param_jk.theta_log, np.array([Param_jk.theta]), axis=0)

        delta_theta_ij_a = 2 * alpha * delta_bw_ij_a * sigmoid_derivative(s_ij_a)
        Param_ij[0].theta += delta_theta_ij_a
        Param_ij[0].theta_log = np.append(Param_ij[0].theta_log, np.array([Param_ij[0].theta]), axis=0)
        
        delta_theta_ij_b = 2 * alpha * delta_bw_ij_b * sigmoid_derivative(s_ij_b)
        Param_ij[1].theta += delta_theta_ij_b
        Param_ij[1].theta_log = np.append(Param_ij[1].theta_log, np.array([Param_ij[1].theta]), axis=0)

        if E_fw < eps:
            correct_list[roop0] = 1
        else:
            correct_list[roop0] = 0

        accuracy = accuracy_checker(correct_list)

        print("accuracy : ", accuracy)

        if accuracy > 0.7:
            x1 = np.arange(-10.0, 10.0, 0.1)
            x2_a = (Param_ij[0].theta - Param_ij[0].W[0] * x1) / Param_ij[0].W[1]
            x2_b = (Param_ij[1].theta - Param_ij[1].W[0] * x1) / Param_ij[1].W[1]
            plot_line = plt.plot(x1, x2_a, label="a")
            plot_line = plt.plot(x1, x2_b, label="b")
            for i in range(Input.size):
                plt_dataset = plt.scatter(Input.X[i,0], Input.X[i,1], c="green")
            plt.title("line")
            plt.xlabel("x1")
            plt.ylabel("x2")
            plt.xlim(-1.5, 2.0)
            plt.ylim(-1.5, 2.0)
            plt.show()

            #learned_flag = True
        
        roop0 += 1
        step += 1

        if roop0 == Input.size / 4:
            epoch += 1
        
        print("roop0 : ", roop0)
        print("step : ", step)
        print("epoch : ", epoch)
        
        if step % int(Input.size/2) == 0 or epoch == 1000:
            plt.cla()
            #plot_W = plt.plot(np.arange(len(Param_jk.W_log)), Param_jk.W_log, label="W")
            x1 = np.arange(-10.0, 10.0, 0.1)
            x2_a = (Param_ij[0].theta - Param_ij[0].W[0] * x1) / Param_ij[0].W[1]
            x2_b = (Param_ij[1].theta - Param_ij[1].W[0] * x1) / Param_ij[1].W[1]
            plot_line = plt.plot(x1, x2_a, label="a")
            plot_line = plt.plot(x1, x2_b, label="b")
            """
            for i in range(Input.size):
                plt_dataset = plt.scatter(Input.X[i,0], Input.X[i,1], c="green")
            """
            plt.title("line")
            plt.xlabel("x1")
            plt.ylabel("x2")
            plt.xlim(-10.0, 10.0)
            plt.ylim(-10.0, 10.0)
            plt.pause(0.00000000000000000001)

        if epoch == 1000:
            break 



"""
    plot_W = plt.plot(np.arange(len(Param_jk.W_log)), Param_jk.W_log, label="W")
    plt.title("W")
    plt.xlabel("step")
    plt.ylabel("W")
    plt.legend()
    plt.show()
"""
