import random
import math
import numpy as np
import matplotlib.pyplot as plt

# input dataset
file_name = "./dataset.txt"

# params
index = 0
attenuation_rate = 0.9
randn_param = 0.001
alpha = 0.05
dimension = 1
x1_c = np.arange(-2, 2, 0.1)
x2_c = np.arange(-2, 2, 0.1)

# declare arrays
X = np.empty((0,2), float)
C_tmp = np.empty((0,2), float)
Cell = np.empty((0,2), float)
Cell_log = np.empty((0, np.empty(0, 2, float)), float)
D = np.empty((0,1), float)
D_log = np.empty((0, np.empty(0, 1, float)), float)
D_threshold = 0.001

try:
    file = open(file_name, "r")
    for line in file:
        if line[0] == "N":
            continue
        else:
            data = line.split()
            x_tmp = np.array([[float(data[1]), float(data[2])]])
            t_tmp = int(data[3])
            w1_init = randn_param*float(np.random.randn(dimension))
            w2_init = randn_param*float(np.random.randn(dimension))
            w_tmp = np.array([[w1_init, w2_init]])
            X = np.append(X, x_tmp, axis=0)
            T.append(t_tmp)
            W = np.append(W, w_tmp, axis=0)
            print("loading now")
            index += 1

except Exception as error:
    print(error)

finally:
    file.close()


def argmin_i(cell_num, j):
    norm_list = np.empty((0,1),float)
    min_i = 0
    min_d = np.linalg.norm(X[j] - Cell[min_i])
    for i in range(cell_num):
        d = np.linalg.norm(X[j] - Cell[i])
        if d < min_d:
            min_i = i
            min_d = d
    return min_i



if __name__ == '__main__':
    learned_flag = False
    correct_answer_rate = 0
    epoch = 0
    cell_num = 0
    flag_D_count = [0 for i in range(cell_num)]
    while learned_flag == False:
        c1 = random.uniform(-1.0, 1.0)
        c2 = random.uniform(-1.0, 1.0)
        C_tmp = np.array([[c1, c2]])
        Cell = np.append(Cell, C_tmp, axis=0)        
        D = np.append(D, 0)
        cell_num += 1
        flag_C_count = [0 for i in range(cell_num)]
        for I in D: #initialize
            D[I] = 0
        convergence_D_num = 0
        indent_flag = False
        while indent_flag == False:
            convergence_Cell_num = 0
            for j in index:
                C = argmin_i(cell_num, j)         
                d = np.linalg.norm(X[j] - Cell[C]) 
                Cell[C] += alpha * (X[j] - Cell[C])
                D[C] = attenuation_rate * D[C] + (1 - attenuation_rate) * d
                Cell_log = np.append(Cell_log[C], Cell[C], axis=0)
                D_log = np.append(D_log[C], D[C], axis=0)
            if np.linalg.norm(Cell_log[C,D[C]] - Cell[C]) == 0:
               flag_C_count[C] = 1
            for k in range(cell_num):
                convergence_Cell_num += flag_count[k]
            if convergence_Cell_num == cell_num:
                indent_flag = True

        for k in range(cell_num):
            if D[k] < D_threshold:
                flag_D_count[k] = 1
        if convergence_D_num == cell_num:
            print("finish")
            learned_flag = True
