import random
import math
import numpy as np
import matplotlib.pyplot as plt
from pyclustering.cluster import xmeans
# input dataset
file_name = "./dataset.txt"

# params
index = 0
attenuation_rate = 0.9
randn_param = 0.1
move_end_threshold = 0.1
D_threshold = 0.5
dimension = 2
alpha = 0.01

X = []
D_log = []


class Cell:
    def __init__(self, position):#, init_xy):
        self.position = position
        self.position_log = np.empty((0,2), float)

try:
    file = open(file_name, "r")
    for line in file:
        if line[0] == "N":
            continue
        else:
            data = line.split()
            X.append([float(data[1]), float(data[2])])
            print("loading now")
            index += 1
except Exception as error:
    print(error)
finally:
    file.close()



if __name__ == '__main__':
    learned_flag = False
    epoch = 0
    cell_num = 0
    cell = []
    min_i = 0
    counter = 0
    roop0 = 0
    
    while learned_flag == False:
        convergence_D_num = 0
        
        position = randn_param * np.random.randn(dimension)# + init w1 and w2
        cell.append(Cell(position))
        cell_num += 1
        cell[-1].position_log = np.append(cell[-1].position_log, [cell[-1].position], axis=0)
        print(cell_num)
        
        D = [0 for i in range(len(cell))]
        
        cell_indent_flag = False
        
        flag_C_count = [0 for i in range(cell_num)]
        flag_D_count = [0 for i in range(cell_num)]
         
        roop1 = 0
        while cell_indent_flag == False:
            convergence_Cell_num = 0
            for j in range(index):
                counter = 0
                #calcurate minimum distance from a data x[j] to cell[*]
                norm_list = np.empty((0,1),float)
                min_i = 0
                min_d = np.linalg.norm(X[j] - cell[min_i].position)
                for i in range(cell_num):
                    d = np.linalg.norm(X[j] - cell[i].position)
                    if d < min_d:
                        min_i = i
                        min_d = d

                #move cell[min_i] to the x[j]
                cell[min_i].position += alpha * (X[j] - cell[min_i].position)
                
                #add log of the moving cell[*] and D
                for i in range(len(cell)):
                    cell[i].position_log = np.append(cell[i].position_log, [cell[i].position], axis=0)
                
                D[min_i] = attenuation_rate * D[min_i] + (1 - attenuation_rate) * min_d
                D_log.append(D[min_i])
                
            #check distance of moving and judge indent_flag
            move_distance = [0 for i in range(cell_num)]
            
            for i in range(cell_num):
                move_distance[i] = np.linalg.norm(cell[i].position_log[-2] - cell[i].position_log[-1])
                if move_distance[i] < move_end_threshold:
                   flag_C_count[i] = 1

            #for i in range(cell_num):
                convergence_Cell_num += flag_C_count[i]
                print("convergence_Cell_num", convergence_Cell_num) 
                if convergence_Cell_num == cell_num:
                    cell_indent_flag = True
                

            roop1 += 1
            #if roop1 == 5:
                #break 
        #check D and judge learned_flag
        for i in range(cell_num):
            if D[i] < D_threshold:
                flag_D_count[i] = 1
            convergence_D_num += flag_D_count[i]
            print("convergence_D_num", convergence_D_num)
            if convergence_D_num == cell_num:
                print("finish")
                learned_flag = True
            
        roop0 += 1
        #if roop0 == 5:
            #break

    plt.figure(1)
    plt.grid(True)
    plt.title("Weight")
    plt.xlabel("x")
    plt.ylabel("y")
    for i in range(index):
        plt.plot(X[i][0],X[i][1],"ok")
    colorlist = ["r", "g", "b", "c", "m"]
    
    for i , j in enumerate(cell):
        size = len(cell[i].position_log)
        print("cell[",i,"].position.size()", size)

    for idx, cell_ in enumerate(cell):
        color = "o"+colorlist[idx]
        plt.plot(cell[idx].position_log[:,0], cell[idx].position_log[:,1], color)
    plt.show()   
