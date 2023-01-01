import os
import math
import numpy as np
import numpy.linalg as la
import pandas as pd
import pickle
from datetime import datetime

gate6CenterWorld = np.array([0.0, 0.0, 2.038498]).reshape(3, )


def generateRandomPose(gateX, gateY, gateZ):
    xmin, xmax = gateX - 4, gateX + 4
    ymin, ymax = gateY - 6, gateY - 9
    zmin, zmax = gateZ - 0.6, gateZ + 2.0
    x = xmin + np.random.rand() * (xmax - xmin)
    y = ymin + np.random.rand() * (ymax - ymin)
    z = zmin + np.random.rand() * (zmax - zmin)
    # maxYawRotation = 55 #25
    # yaw = np.random.normal(90, maxYawRotation/5) # 99.9% of the samples are in 5*segma
    minYaw, maxYaw = 90-35, 90+35
    yaw = minYaw + np.random.rand() * (maxYaw - minYaw)
    return x, y, z, yaw


def main():
    # np.random.seed(0)

    benchmarkPosesRootDir = '/home/majd/catkin_ws/src/basic_rl_agent/data/deep_learning/benchmarks/benchmarkPosesFiles'
    # fileName = 'benchmarkerPosesFile_#5_202205211439_14.pkl'
    # posesDataFrame = pd.read_pickle(os.path.join(benchmarkPosesRootDir, fileName))
    # poses = np.array(posesDataFrame['poses'].tolist())

    rhoMean = 2  # [m/s]
    rhoMax = 4
    rhoStd = (rhoMax-rhoMean) / 5  # [m/s]
    thetaStd = (np.pi/2)/5
    psiStd = (np.pi/5)/5

    accNonZero = True
    accMax = 5  # [m/s^2]

    nPoses = 50

    poseVelList = []
    gateX, gateY, gateZ = gate6CenterWorld
    for i in range(nPoses):

        droneX, droneY, droneZ, droneYaw = generateRandomPose(
            gateX, gateY, gateZ)

        randRho = np.random.normal(rhoMean, rhoStd)
        randTheta = np.random.normal(0, thetaStd)
        randPsi = np.random.normal(0, psiStd)

        randX = randRho*np.cos(randPsi)*np.cos(randTheta)
        randY = randRho*np.cos(randPsi)*np.sin(randTheta)
        randZ = randRho*np.sin(randPsi)

        # applying 90 rotaiton on the Z axis:
        randX, randY = -randY, randX

        poseVel = np.array(
            [droneX, droneY, droneZ, droneYaw, randX, randY, randZ, 0])

        if accNonZero:
            acc = np.array([randX, randY, randZ, 0])
            acc = (acc / la.norm(acc)) * np.random.rand() * accMax
            poseVel = np.concatenate([poseVel, acc])
        poseVelList.append(poseVel)

    df = pd.DataFrame({'poses': poseVelList})
    fileName = "benchmarkerPosesFile_nonStationary{}_#{}_{}.pkl".format(
        'Acc' if accNonZero else '', nPoses, datetime.now().strftime("%Y%m%d-%H%M%S"))
    df.to_pickle(os.path.join(benchmarkPosesRootDir, fileName))
    print("{} was saved!".format(fileName))

    # with open(os.path.join(benchmarkPosesRootDir, fileName),'wb') as f:
    #     pickle.dump(poseVel, f)


if __name__ == '__main__':
    main()
