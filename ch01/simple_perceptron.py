import random
import math
import numpy as np
import matplotlib.pyplot as plt

# input dataset
file_name = "./dataset.txt"

alpha = 0.05
theta = random.uniform(-0.01, 0.01)

X = np.array([])
W = np.array([])
T = np.array([])


index = 0

try:
    file = open(file_name, "r")
    w_tmp = [random.uniform(-0.01, 0.01), random.uniform(-0.01, 0.01)]
    for line in file:
        if line[0] == "N":
            continue
        else:
            data = line.split()
            x_tmp = [float(data[1]), float(data[2])]
            t_tmp = float(data[3])
            np.append(X, x_tmp)
            np.append(T, t_tmp)
            np.append(W, w_tmp)
            index += 1

except Exception as error:
    print(error)

finally:
    file.close()


def calc_phi(num):
    if num > 0:
        return 1
    else:
        return 0


def calc_output(x1, x2, w1, w2):
    sigma_wx = x1 * w1 + x2 * w2
    output = calc_phi(sigma_wx - theta)
    return output



if __name__ == '__main__':
    learned_flag = False
    correct_count = 0
    correct_answer_rate = 0
    i = 0

    while learned_flag == False:
        i = i % index
        x1_tmp = X[i:0]
        x2_tmp = X[i:1]
        w1_tmp = W[i:0]
        w2_tmp = W[i:1]
        y = calc_output(x1_tmp, x2_tmp, w1_tmp, w2_tmp)

        if y == T[0:i]:
            correct_count += 1
        else:
            delta = y - T[0:i]
            W[i:0] -= alpha * delta * X[i:0]
            W[i:1] -= alpha * delta * X[i:1]
            theta += alpha * delta
    

        if i == index - 1:
            correct_answer_rate = correct_count / index
            if correct_answer_rate == 1:
                print("finish!!!")
                learned_flag = True
            else:
                print(i)
                print("rate : ", correct_answer_rate * 100)
            
        i += 1
        print(i)


