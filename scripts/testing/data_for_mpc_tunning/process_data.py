import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


def main():
    df = pd.read_pickle('./data_for_mpc_tunning/data.pkl')
    time = df['time'].tolist()
    rollData = df['roll'].tolist()

    time = np.array(time, dtype=np.float64)
    rollData = np.array(rollData, dtype=np.float64)
    # plt.plot(time, rollData)
    # plt.show()
    minTimeNotZero = time[rollData > 0.15]
    print(minTimeNotZero[0])


    oneTau = 1-math.exp(-1)
    rollData = rollData/20
    idx = (np.abs(rollData - oneTau)).argmin()
    print(idx, time[idx])
    tau = time[idx] - minTimeNotZero[0]
    print('tau = {}'.format(tau))
    plt.plot(time, rollData)
    plt.axvline(x=minTimeNotZero[0])
    plt.axvline(x=time[idx])
    plt.show()
    
    


if __name__=="__main__":
    main()