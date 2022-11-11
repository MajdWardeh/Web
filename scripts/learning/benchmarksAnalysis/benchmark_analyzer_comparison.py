from distutils.command.config import config
from doctest import SkipDocTestCase
import os
import math
import numpy as np
from numpy import linalg as la
import pandas as pd
import pickle
import matplotlib.pyplot as plt



def compareBechmarkResults(resultsRoot, dictFile1, dictFile2):
    resDict1 = pd.read_pickle(os.path.join(resultsRoot, dictFile1))
    resDict2 = pd.read_pickle(os.path.join(resultsRoot, dictFile2))

    startingPosesList1 = resDict1['pose']
    startingPosesList2 = resDict2['pose']

    finishReasonList1 = resDict1['round_finish_reason']
    finishReasonList2 = resDict2['round_finish_reason']

    commonStartingPosesList = []

    posesList1, posesList2 = [], []
    twistList1 = []
    twistList2 = []
    accList1, accList2 = [], []


    assert 'traversingTime' in resDict1.keys()
    assert 'traversingTime' in resDict2.keys()

    sucCount1, sucCount2 = 0, 0
    skipCount1, skipCount2 = 0, 0
    for idx, pose1 in enumerate(startingPosesList1):
        if finishReasonList1[idx] == 'dronePassedGate' or finishReasonList1[idx] == 'droneInFrontOfGate':
            sucCount1 += 1
        elif finishReasonList1[idx] == 'bad pose, skipped':
            skipCount1 += 1
    for idx, pose1 in enumerate(startingPosesList1):
        if finishReasonList2[idx] == 'dronePassedGate' or finishReasonList2[idx] == 'droneInFrontOfGate':
            sucCount2 += 1
        elif finishReasonList2[idx] == 'bad pose, skipped':
            skipCount2 += 1

    Tlist1, Tlist2 = [], []
    traverseTimeDiffList = []

    

    for idx, pose1 in enumerate(startingPosesList1):
        if (pose1 != startingPosesList2[idx]).any():
            # print(pose1, posesList2[idx])
            continue
        
        if finishReasonList1[idx] == 'dronePassedGate' and finishReasonList2[idx] == 'dronePassedGate':
            commonStartingPosesList.append(pose1)

            posesList1.append(resDict1['posesList'][idx])
            twistList1.append(resDict1['twistList'][idx])
            accList1.append(resDict1['linearAccList'][idx])

            posesList2.append(resDict1['posesList'][idx])
            twistList2.append(resDict2['twistList'][idx])
            accList2.append(resDict2['linearAccList'][idx])

            t1 = resDict1['traversingTime'][idx].to_sec()
            t2 =  resDict2['traversingTime'][idx].to_sec()
            Tlist1.append(t1)
            Tlist2.append(t2)
            traverseTimeDiffList.append(t2-t1)
            # print(resDict1['traversingTime'][idx].to_sec(), resDict2['traversingTime'][idx].to_sec(), t)


    print('success rates: {}, {}'.format(sucCount1/(100-skipCount1), sucCount2/(100-skipCount2)))
    avgVelList1 = np.array([la.norm(vel[:, :-1], axis=1).mean() for vel in twistList1])
    avgVelList2 = np.array([la.norm(vel[:, :-1], axis=1).mean() for vel in twistList2])
    print(avgVelList1.mean(), avgVelList2.mean())
    print(avgVelList1.min(), avgVelList1.max(), avgVelList2.min(), avgVelList2.max())

    traverseTimeDiffList = np.array(traverseTimeDiffList)
    print('MeanTraversingTimeDiff: ', traverseTimeDiffList.mean())


    from scipy import stats
    # print(stats.ttest_ind(avgVelList1, avgVelList2, equal_var=False))
    print("stats analysis:", stats.ttest_ind(Tlist1, Tlist2, equal_var=False))
    count = 0
    for vel1, vel2 in zip(avgVelList1, avgVelList2):
        if vel1 > vel2:
            count += 1
    print(avgVelList1.shape, count)
    print(avgVelList1.min(), avgVelList1.max())

    maxVelList1 = np.array([vel[:, :-1].max() for vel in twistList1])
    maxVelList2 = np.array([vel[:, :-1].max() for vel in twistList2])

    print('maxVel Computing:')
    print('maxVel mean: ', maxVelList1.mean(), maxVelList2.mean())
    print('maxVel peak:', maxVelList1.max(), maxVelList2.max())
    print('maxVel peak idx:', maxVelList1.argmax(), maxVelList2.argmax())
    peakVel1 = maxVelList1.argmax()
    speedArrayForPeakCase1_m1 = twistList1[peakVel1][:, :-1] # remove the yaw rate
    posesListPeakCase1_m1 = posesList1[peakVel1]
    speedArrayForPeakCase1_m2 = twistList2[peakVel1][:, :-1] # remove the yaw rate

    print('posesListPeakCase1_m1.shape', posesListPeakCase1_m1.shape)

    fig, axes = plt.subplots(4, 1)
    axes[0].plot(np.arange(len(speedArrayForPeakCase1_m1)), speedArrayForPeakCase1_m1)
    axes[1].plot(np.arange(len(speedArrayForPeakCase1_m1)), la.norm(speedArrayForPeakCase1_m1, axis=1), 'b')
    axes[1].plot(np.arange(len(speedArrayForPeakCase1_m2)), la.norm(speedArrayForPeakCase1_m2, axis=1), 'r')
    axes[2].plot(posesListPeakCase1_m1[:, 0]-posesListPeakCase1_m1[0, 0],  posesListPeakCase1_m1[:, 1:-1])
    axes[3].plot(np.arange(len(posesListPeakCase1_m1[:, 0])), posesListPeakCase1_m1[:, 0])
    plt.show()


    commonStartingPosesList = np.array(commonStartingPosesList)
    print(commonStartingPosesList[:, -1].mean(), commonStartingPosesList[:, -1].std())
    for i in range(4):
        print(i, commonStartingPosesList[:, i].min(), commonStartingPosesList[:, i].max())
    exit()
    # plt.hist(comPosesList[:, -1], bins='auto')
    # plt.show()
    # exit()

    zerosYawList = np.abs(commonStartingPosesList[:, -1] - 90)
    print(zerosYawList.min(), zerosYawList.max())

    yawRateList1 = [vel[:, -1] for vel in twistList1]
    yawRateList2 = [vel[:, -1] for vel in twistList2]

    for _ in range(len(yawRateList1)):
        rand = np.random.randint(0, len(yawRateList1))
        print(rand)
        y1 = yawRateList1[rand]
        y2 = yawRateList2[rand]
        x1 = np.arange(0, y1.shape[0])
        x2 = np.arange(0, y2.shape[0])
        plt.plot(x1, y1, 'b')
        plt.plot(x2, y2, 'r')
        plt.show()

    

    exit()


    peakVelList1 = np.array([v.max() for v in twistList1])
    peakVelList2 = np.array([v.max() for v in twistList2])

    print(peakVelList1.max())
    print(peakVelList2.max())
    

def compareFM(benchmarksResultsDir, frameModeList, addMore=False, filterList=[], minFrameMode=5):
    meanTraverseTimeList = []
    successRateList = []
    skippedList = []

    if addMore:
        for file in os.listdir(benchmarksResultsDir):
            result = np.array([(f in file) for f in filterList])
            if result.all():
                for i in range(minFrameMode, 60):
                    if 'frameMode{}'.format(i) in file:
                        frameModeList.append(file)

    frameModeFileDict = {}
    for i in range(60):
        for file in frameModeList: 
            if 'frameMode{}'.format(i) in file:
                frameModeFileDict[i] = file

    frameModeSuccessRateDict = {}
    frameModeDictDict = {}
    for i, file in frameModeFileDict.items():
        try:
            print('trying to open {}'.format(file))
            dict_i = pd.read_pickle(os.path.join(benchmarksResultsDir, file))
            frameModeDictDict[i] = dict_i
        except Exception as e:
            print(e)
            continue

        traverseTimeList = []
        successCount = 0
        skippedCount = 0
        for idx, _ in enumerate(dict_i['pose']):
            if dict_i['round_finish_reason'][idx] == 'bad pose, skipped':
                skippedCount += 1
                continue
            if dict_i['round_finish_reason'][idx] == 'dronePassedGate' or dict_i['round_finish_reason'][idx] == 'droneInFrontOfGate':
                successCount+=1 
                t = dict_i['traversingTime'][idx]
                traverseTimeList.append(t.to_sec())

        frameModeSuccessRateDict[i] = int(round(successCount*100/(100-skippedCount) ))

        meanTraverseTime = np.array(traverseTimeList).mean()
        meanTraverseTimeList.append(meanTraverseTime)
    
    with plt.style.context(['science', 'ieee', 'std-colors']):
        fig, ax = plt.subplots()
        x = list(frameModeSuccessRateDict.keys())
        x.sort()
        y = [frameModeSuccessRateDict[i] for i in x]
        print(x)
        print(y)
        ax.plot(x, y, label='our method')
        ax.legend()
        ax.autoscale(tight=True)
        # ax.set(**pparam)
        # Note: $\mu$ doesn't work with Times font (used by ieee style)
        ax.set_ylim([0, 100])
        ax.set_ylabel(r'number of successful traverses')  
        # fig.savefig('figures/fig2b.pdf')
        # fig.savefig('figures/fig2b.jpg', dpi=300)
        plt.show()
    # print(meanTraverseTimeList)
    # print(successRateList)
    # print(skippedList)
        
    

        

def compareWithBaseline():
    benchmarksResultsDir = '/home/majd/catkin_ws/src/basic_rl_agent/data/deep_learning/benchmarks/results'
    # dictRes1 = 'config17_BeizerLoss_imageToBezierData1_1800_20210905-1315_benchmarkerPosesFile_#100_202109052231_28_frameMode1_202208121850_05.pkl'
    # dictRes1 = 'config17_BeizerLoss_imageToBezierData1_1800_20210905-1315_benchmarkerPosesFile_#100_202205081959_38_frameMode1_202208162307_43.pkl'
    dictRes1 = 'config17_BeizerLoss_imageToBezierData1_1800_20210905-1315_benchmarkerPosesFile_#100_202205081959_38_frameMode1_202208162307_43.pkl'

    dictRes2 = 'rpg_sim2real_test_benchmark_benchmarkerPosesFile_#100_202205081959_38_frameMode1_202208170953_22.pkl'
    # dictRes2 = 'rpg_sim2real_test_benchmark_benchmarkerPosesFile_#100_202205081959_38_modified_frameMode1_202208172018_09.pkl'
    compareBechmarkResults(benchmarksResultsDir, dictRes1, dictRes2)

def compareWithFrameMode():
    benchmarksResultsDir = '/home/majd/catkin_ws/src/basic_rl_agent/data/deep_learning/benchmarks/results'
    frameModeList = [
        "config17_BeizerLoss_imageToBezierData1_1800_20210905-1315_benchmarkerPosesFile_#100_202205081959_38_frameMode1_202208162307_43.pkl",
        "config17_BeizerLoss_imageToBezierData1_1800_20210905-1315_benchmarkerPosesFile_#100_202205081959_38_frameMode2_202208170001_42.pkl",
        "config17_BeizerLoss_imageToBezierData1_1800_20210905-1315_benchmarkerPosesFile_#100_202205081959_38_frameMode3_202208171115_23.pkl",
        "config17_BeizerLoss_imageToBezierData1_1800_20210905-1315_benchmarkerPosesFile_#100_202205081959_38_frameMode4_202208171145_33.pkl",
        # "config17_BeizerLoss_imageToBezierData1_1800_20210905-1315_benchmarkerPosesFile_#100_202205081959_38_frameMode5_202208171211_25.pkl",
        # "config17_BeizerLoss_imageToBezierData1_1800_20210905-1315_benchmarkerPosesFile_#100_202205081959_38_frameMode6_202208171250_59.pkl",
        # "config17_BeizerLoss_imageToBezierData1_1800_20210905-1315_benchmarkerPosesFile_#100_202205081959_38_frameMode7_202208171745_55.pkl",
        # "config17_BeizerLoss_imageToBezierData1_1800_20210905-1315_benchmarkerPosesFile_#100_202205081959_38_frameMode8_202208171830_00.pkl",
        # "config17_BeizerLoss_imageToBezierData1_1800_20210905-1315_benchmarkerPosesFile_#100_202205081959_38_frameMode10_202208172211_15.pkl",
        # "config17_BeizerLoss_imageToBezierData1_1800_20210905-1315_benchmarkerPosesFile_#100_202205081959_38_frameMode12_202208172302_12.pkl",
        # "config17_BeizerLoss_imageToBezierData1_1800_20210905-1315_benchmarkerPosesFile_#100_202205081959_38_frameMode14_202208172335_31.pkl",
        # "config17_BeizerLoss_imageToBezierData1_1800_20210905-1315_benchmarkerPosesFile_#100_202205081959_38_frameMode16_202208180006_24.pkl",
    ]
    # compareFM(benchmarksResultsDir, frameModeList, addMore=True, filterList=['config17', '202208'], minFrameMode=7)
    compareFM(benchmarksResultsDir, frameModeList, addMore=True, filterList=['rpg', '202208'], minFrameMode=2)

        

if __name__ == '__main__':
    compareWithBaseline()
    # compareWithFrameMode()