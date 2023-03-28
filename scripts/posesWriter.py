import os
import numpy as np
import pandas as pd
import random


def main():
    benchmarkPosesRootDir = '/home/majd/catkin_ws/src/basic_rl_agent/data/deep_learning/benchmarks/benchmarkPosesFiles'
    fileName = 'benchmarkerPosesFile_#100_202205081959_38_copy.pkl'

    targetPosesHumandNumbers = [40, 51, 65, 97, 4, 7, 9, 11, 33, 37, 39,
                                42, 48, 50, 54, 55, 57, 58, 61, 63, 67, 76, 79, 84, 87, 91, 92, 98]

    targetPosesHumandNumbers = [40, 51, 65, 97, 4, 7, 9, 11, 33, 37, 39,
                                42, 48, 50, 54, 55, 57, 58, 61, 63, 67, 76, 79, 84, 87, 91, 92, 98]

    # targetPosesHumandNumbers = random.sample(targetPosesHumandNumbers, 15)


    targetPosesIdices = [i - 1 for i in targetPosesHumandNumbers]


    posesDataFrame = pd.read_pickle(
        os.path.join(benchmarkPosesRootDir, fileName))
    poses = posesDataFrame['poses'].tolist()
    targetPoses = [poses[i] for i in targetPosesIdices]


    goodPoses = [2, 1, 3, 6, 7 ,8 , 9, 11, 12, 13, 14, 15]
    targetPoses = [targetPoses[i] for i in goodPoses]


    df = pd.DataFrame({'poses': targetPoses})
    df_name = 'benchmarkerPosesFile_#100_202205081959_38E_filtered{}.pkl'.format(len(targetPoses))
    df.to_pickle(os.path.join(benchmarkPosesRootDir, df_name))
    print('file: {} saved'.format(os.path.join(benchmarkPosesRootDir, df_name)))


if __name__ == '__main__':
    main()
