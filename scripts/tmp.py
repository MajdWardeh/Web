import numpy as np
import pandas as pd
from scipy.special import binom


def main():
    weightsFile = '/home/majd/catkin_ws/src/basic_rl_agent/data/deep_learning/MarkersToBezierDataFolder/models_weights/weights_MarkersToBeizer_FC_scratch_withYawAndTwistData_config19_20210827-060041.h5'
    configNum = 'config19'

    startIndex = weightsFile.find(configNum, 0)
    assert startIndex != -1
    fileName = weightsFile[startIndex:].split('.')[0]
    print(fileName)

if __name__ == '__main__':
    main()
