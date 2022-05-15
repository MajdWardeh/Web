import time
import numpy as np
import pandas as pd
from MarkersToBezierGenerator import  MarkersAndTwistDataToBeizerDataGenerator



def main():
    path = '/home/majd/catkin_ws/src/basic_rl_agent/data2/flightgoggles/datasets/imageBezierDataV2_1_1000/allData_WITH_STATES_PROB_filtered_imageBezierDataV2_1_1000_20220416-1501.pkl' 
    save_path = '/home/majd/catkin_ws/src/basic_rl_agent/data2/flightgoggles/datasets/imageBezierDataV2_1_1000/allData_WITH_STATES_PROB_filtered2_imageBezierDataV2_1_1000_20220416-1501.pkl' 
    df = pd.read_pickle(path)
    df = df.astype(np.float32)
    df.to_pickle(save_path)
    print(df.columns)
    print('sleeping...')
    time.sleep(20)

    # print('waked up')
    # inputImageShape = (480, 640, 3) 
    # train_Xset, train_Yset = [df['markersData'].tolist(), df['vel'].tolist()], [df['positionControlPoints'].tolist(), df['yawControlPoints'].tolist()]
    # statesProbList = df['statesProbList'] 
    # trainGenerator = MarkersAndTwistDataToBeizerDataGenerator(train_Xset, train_Yset, 1024, inputImageShape, statesProbList=statesProbList)



if __name__ == '__main__':
    main()