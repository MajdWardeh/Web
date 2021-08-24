import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import time

def get_Timage():
    file1 = open('left_images.txt', 'r')
    Lines = file1.readlines()
    data = []
    head = Lines[0]
    for line in Lines[1:]:
        line_data = line.split(' ')
        data.append(line_data[:-1])
    file1.close()
    data = np.array(data)
    time_diff = []
    for index, t in enumerate(data[:-1, 1]):
        time_diff.append(float(data[index+1, 1])-float(t) )
    time_diff = np.array(time_diff)
    return time_diff.mean()

def getData():
    file1 = open('groundtruth.txt', 'r')
    Lines = file1.readlines()
    data = []
    head = Lines[0]
    # print("data head: {}".format(head))
    for line in Lines[1:]:
        line_data = line.split(' ')
        data.append(line_data)
    file1.close()
    data = np.array(data, dtype=float)
    return data

def get_Tpose(data):
    dts = []
    for i, t in enumerate(data[:-1, 1]):
        dts.append(data[i+1, 1]-t)
    dts = np.array(dts)
    return dts.mean()    

def optimize_curve(data, axis):
    t = data[:, 0]
    t = t - t[0]
    t = t[:, np.newaxis].astype(np.float64)
    b = (data[:, axis] - data[0, axis]).astype(np.float64)
    A = np.concatenate((np.ones(t.shape, dtype=np.float64), t, np.square(t), np.power(t, 3), np.power(t, 4)), axis=1)
    
    n = 5
    x = cp.Variable(n)
    objective = cp.Minimize(cp.sum_squares(A@x - b))
    prob = cp.Problem(objective)
    result = prob.solve(warm_start=True)
    # print(t.shape)
    # print(result)
    # print(x.value)
    # plt.plot(t, b, 'b')
    # plt.plot(t, A@x.value, 'r')
    # plt.show()
    return result, x.value

    
if __name__ == "__main__":
    Timage = get_Timage()
    data = getData()
    Tpose = get_Tpose(data)
    c = int(2*Timage/Tpose)
    
    t_start = time.time()
    for axis in range(1, 4):
        costs = []
        for i, t in enumerate(data[:-c, 1]):
            data_dTimage = data[i:i+c, 1:5]
            cost, _ = optimize_curve(data_dTimage, axis)
            costs.append(cost)
        t_end = time.time()
        costs = np.array(costs)
        print("axis: {}, max: {}, argmax: {}, time: {}".format(axis, costs.max(), costs.argmax(), t_end-t_start))
