import sys
import random
import math
import numpy as np
import matplotlib.pyplot as plt

# input dataset
file_name = "./dataset.txt"

alpha = 0.05
random_min = 0.1
dimension = 1
theta = np.random.randn(dimension)
#w_tmp = np.array((0,2), float)

X = np.empty((0,2), float)
W = np.empty((0,2), float)
# T = np.empty((0,1), int)
T = []
W_log = np.empty((0,2), float)
theta_log = np.empty((0,2), float)

index = 0

print("input data")

try:
    file = open(file_name, "r")
    for line in file:
        if line[0] == "N":
            continue
        else:
            data = line.split()
            x_tmp = np.array([[float(data[1]), float(data[2])]])
            t_tmp = int(data[3])
            w1_init = float(np.random.randn(dimension))
            w2_init = float(np.random.randn(dimension))
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




def calc_phi(num):
    if num > 0:
        #print("1")
        return 1
    else:
        return 0


def calc_output(x1, x2, w1, w2):
    sigma_wx = x1 * w1 + x2 * w2
    #print("sigma:", sigma_wx)
    output = calc_phi(sigma_wx - theta)
    return output


learn_count = 0

if __name__ == '__main__':
    learned_flag = False
    correct_count = 0
    correct_answer_rate = 0
    i = 0
    while learned_flag == False:
        i = i % index
        x1_tmp = X[i, 0]
        x2_tmp = X[i, 1]
        w1_tmp = W[i, 0]
        w2_tmp = W[i, 1]
        y = calc_output(x1_tmp, x2_tmp, w1_tmp, w2_tmp)
        
        if y == T[i]:
            print("correct!")
            correct_count += 1
        else:
            print("learn!")
            delta = y - T[i]
            W[i, 0] -= alpha * delta * X[i, 0]
            W[i, 1] -= alpha * delta * X[i, 1]
            theta += alpha * delta
            learn_count += 1
        
        W_log_tmp = np.array([[W[i,0], W[i,1]]])
        W_log = np.append(W_log, W_log_tmp, axis=0)
        

        if i == index - 1:
            correct_answer_rate = correct_count / index
            correct_count = 0
            if correct_answer_rate == 1:
                print("finish!!!")
                learned_flag = True
                print("learn count : ", learn_count)
            else:
                print("rate : ", correct_answer_rate * 100)
            
        i += 1
        # print(i)


step = np.arange(learn_count)

plots = plt.plot(step, W_log[:,0], label="w1")
plt.title("simple_perceptron")
plt.xlabel("step")
plt.ylabel("w1")
plt.legend()
plt.show()

