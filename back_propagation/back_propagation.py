import random
import math
import numpy as np
import matplotlib.pyplot as plt

# input dataset
file_name = "./dataset.txt"

randn_param = 0.001
alpha = 0.05
dimension = 1
theta = randn_param*np.random.randn(dimension)
#w_tmp = np.array((0,2), float)

x1_c = np.arange(-2, 2, 0.1)
x2_c = np.arange(-2, 2, 0.1)

X = np.empty((0,2), float)
W = np.empty((0,2), float)
W_log = np.empty((0,2), float)
W1W2_log = np.empty((0,1), float)
theta_log = np.empty((0,1), float)
thetaW2_log = np.empty((0,1), float)
accuracy_log = np.empty((0,1), float)
accuracy_log_epoch = np.empty((0,1), float)
T = []

index = 0

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
            #print("X:", X)
            T.append(t_tmp)
            #print("T:", T)
            W = np.append(W, w_tmp, axis=0)
            #print("W:", W)
            print("loading now")
            index += 1

except Exception as error:
    print(error)

finally:
    file.close()



def sigmoid(x):
    y = 1 / (1 + np.exp(-a*x))
    return y


if __name__ = '__main__':

