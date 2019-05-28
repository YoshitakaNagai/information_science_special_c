import random
import math
import numpy as np
import matplotlib.pyplot as plt

# input dataset
file_name = "./dataset.txt"

randn_param = 0.1
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

w1_init = randn_param*float(np.random.randn(dimension))
w2_init = randn_param*float(np.random.randn(dimension))


try:
    file = open(file_name, "r")
    for line in file:
        if line[0] == "N":
            continue
        else:
            data = line.split()
            x_tmp = np.array([[float(data[1]), float(data[2])]])
            t_tmp = int(data[3])
            X = np.append(X, x_tmp, axis=0)
            T.append(t_tmp)
            print("loading now")
            index += 1

except Exception as error:
    print(error)

finally:
    file.close()


correct_idx = [0] * index


def calc_phi(num):
    if num > 0:
        return 1
    else:
        return 0


def calc_output(x1, x2, w1, w2):
    sigma_wx = x1 * w1 + x2 * w2
    output = calc_phi(sigma_wx - theta)
    return output


def calc_accuracy():
    correct_count = 0
    for i in range(index):
        if correct_idx[i] == 1:
            correct_count += 1
    return correct_count / index


learn_count = 0
roop = 0

if __name__ == '__main__':
    learned_flag = False
    correct_answer_rate = 0
    i = 0
    idx = np.arange(index)
    k = 0
    epoch = 0
    w1 = w1_init
    w2 = w2_init

    while learned_flag == False:
        k = k % index
        i = int(np.random.choice(idx, 1, replace=False))
        x1_tmp = X[i, 0]
        x2_tmp = X[i, 1]
        y = calc_output(x1_tmp, x2_tmp, w1, w2)
        
        if y == T[i]:
            #print("correct!")
            correct_idx[i] = 1

        else:
            #print("learn!")
            delta = y - T[i]
            w1 -= alpha * delta * x1_tmp
            w2 -= alpha * delta * x2_tmp
            theta += alpha * delta
            learn_count += 1
       
        W_log_tmp = np.array([[w1, w2]])
        W1W2_log_tmp = np.array([[w1/w2]])
        W_log = np.append(W_log, W_log_tmp, axis=0)
        W1W2_log = np.append(W1W2_log, W1W2_log_tmp, axis=0)
        theta_log = np.append(theta_log, theta/w2)
        thetaW2_log = np.append(thetaW2_log, theta)
        accuracy_log = np.append(accuracy_log, correct_answer_rate)

        correct_answer_rate = calc_accuracy()
        
        
        print("rate : ", correct_answer_rate * 100, "%")
        if correct_answer_rate == 1:
            print("finish!!!")
            print("learn count : ", roop)
            print("epoch : ", epoch)
            print("w1_init", w1_init)
            print("w2_init", w2_init)
            learned_flag = True
            if w2 != 0:
                x2_c = theta/w2 - w1 * x1_c / w2
                plot_borderline = plt.plot(x1_c, x2_c, label="classification_line")
                #plot_dataset = plt.scatter(X[:,0], X[:,1], label="dataset")
                for j in range(index):
                    if T[j] == 0:
                        plot_dataset = plt.scatter(X[j,0], X[j,1], c="red")
                    else:
                        plot_dataset = plt.scatter(X[j,0], X[j,1], c="blue")
                plt.title("final_classification_line")
                plt.xlabel("x1")
                plt.ylabel("x2")
                plt.legend()
                plt.xlim(-1.0, 2.0)
                plt.ylim(-1.0, 2.0)
                plt.show()
 
        if k == index - 1:
            idx = np.arange(index)
            
            correct_answer_rate = calc_accuracy()
            accuracy_log_epoch = np.append(accuracy_log_epoch, correct_answer_rate)
            
            if w2 != 0:
                x2_c = theta/w2 - w1 * x1_c / w2
                plot_borderline = plt.plot(x1_c, x2_c, label="classification_line")
                for j in range(index):
                    if T[j] == 0:
                        plot_dataset = plt.scatter(X[j,0], X[j,1], c="red")
                    else:
                        plot_dataset = plt.scatter(X[j,0], X[j,1], c="blue")
                plt.title("classification_line")
                plt.xlabel("x1")
                plt.ylabel("x2")
                plt.legend()
                plt.xlim(-1.0, 2.0)
                plt.ylim(-1.0, 2.0)
                plt.show()

            epoch += 1
             
        k += 1
        roop += 1


plot_w1 = plt.plot(np.arange(roop), W_log[:,0], label="w1")
plt.title("w1")
plt.xlabel("step")
plt.ylabel("w1")
plt.legend()
plt.show()

plot_w2 = plt.plot(np.arange(roop), W_log[:,1], label="w2")
plt.title("w2")
plt.xlabel("step")
plt.ylabel("w2")
plt.legend()
plt.show()

plot_theta = plt.plot(np.arange(roop), theta_log, label="theta")
plt.title("theta")
plt.xlabel("step")
plt.ylabel("theta")
plt.legend()
plt.show()

plot_theta = plt.plot(np.arange(roop), W1W2_log, label="w1/w2")
plt.title("w1/w2")
plt.xlabel("step")
plt.ylabel("w1/w2")
plt.legend()
plt.show()

plot_theta = plt.plot(np.arange(roop), thetaW2_log, label="theta/w2")
plt.title("theta/w2")
plt.xlabel("step")
plt.ylabel("theta/w2")
plt.legend()
plt.show()

plot_accuracy = plt.plot(np.arange(roop), accuracy_log, label="accuracy")
plt.title("accuracy")
plt.xlabel("step")
plt.ylabel("accuracy")
plt.legend()
plt.show()

plot_accuracy = plt.plot(np.arange(epoch), accuracy_log_epoch, label="accuracy")
plt.title("accuracy")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend()
plt.show()

