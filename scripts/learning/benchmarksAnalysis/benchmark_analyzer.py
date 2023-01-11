import os
import math
import numpy as np
from numpy import linalg as la
import pandas as pd
import pickle
import matplotlib.pyplot as plt

class BenchmarkAnalyzer:

    def __init__(self, benchmarksResultsDir, configNumsList=None):
        self.bechmarkResultsDir = benchmarksResultsDir
        self.ListOfDicts = []
        plot_file = False

        if configNumsList is None:
            for file in os.listdir(benchmarksResultsDir)[:]:
                self.processBechmarkResultFile(file)
        else:
            for file in os.listdir(benchmarksResultsDir)[:]:
                for configNum in configNumsList:
                    if configNum in file:
                        plot_file = configNum==file
                        self.processBechmarkResultFile(file, plot_file=plot_file)
        
        if not plot_file:
            sortedListOfDicts = sorted(self.ListOfDicts, key=lambda x: x['numOfSuccesses'], reverse=True)
            for dict in sortedListOfDicts:
                print(dict['fileName'], dict['numOfSuccesses'], dict['avergeSpeed'], dict['averagePeakSpeed'], dict['MaxPeakSpeed'] , dict['skippedPoses'])
 

    def processBechmarkResultFile(self, file, plot_file=False):
        resDict = pd.read_pickle(os.path.join(self.bechmarkResultsDir, file))

        numOfSuccesses = 0
        peakSpeeedList = []
        avergeSpeedList = []
        linearSpeedList = []
        linearAccList = []
        skipped_count = 0
        for idx, finish_reason in enumerate(resDict['round_finish_reason']):
            if finish_reason == 'dronePassedGate' or finish_reason == 'droneInFrontOfGate':
                numOfSuccesses += 1
                if finish_reason == 'dronePassedGate':
                    peak = resDict['peak_twist'][idx]
                    avg = resDict['average_twist'][idx]
                    peakSpeeedList.append(peak)
                    avergeSpeedList.append(avg)
                    if 'linearVelList' in resDict.keys():
                        linearSpeed = resDict['linearVelList'][idx]
                        linearSpeedList.append(linearSpeed)
                    if 'linearAccList' in resDict.keys():
                        linearAcc = resDict['linearAccList'][idx]
                        linearAccList.append(linearAcc)
            if finish_reason == 'bad pose, skipped':
                skipped_count += 1
                # print('found skipped pose, count: {}'.format(skipped_count))



        avergeSpeedList = np.array(avergeSpeedList)

        averagePeakSpeed = np.mean(peakSpeeedList) if len(peakSpeeedList) != 0 else math.nan
        maxPeakSpeed = np.max(peakSpeeedList) if len(peakSpeeedList) != 0 else math.nan
        peakSpeed = np.max(peakSpeeedList) if len(peakSpeeedList) != 0 else math.nan
        averageSpeed = la.norm(avergeSpeedList.reshape(-1, 4)[:-1], axis=1).mean() if len(avergeSpeedList) != 0 else math.nan

        # if 'linearVelList' in resDict.keys():
        #     linearSpeedList = np.array(linearSpeedList)
        # if 'linearAccList' in resDict.keys():
        #     linearAccList = np.array(linearAccList)

        resDict['numOfSuccesses'] = numOfSuccesses
        resDict['averagePeakSpeed'] = averagePeakSpeed
        resDict['MaxPeakSpeed'] = maxPeakSpeed
        resDict['avergeSpeed'] = averageSpeed
        resDict['fileName'] = file
        resDict['skippedPoses'] = skipped_count

        # poses = np.array(resDict['pose'])
        # print(poses.shape)
        # print(poses.min(axis=0), poses.max(axis=0), poses.std(axis=0))



        self.ListOfDicts.append(resDict)

        if plot_file:
            print('processing file: {}'.format(file))
            print('numOfSuccesses: {}, numOfSkippedPoses: {}'.format(numOfSuccesses, skipped_count))
            print('averagePeackSpeed: {}, peak: {}, averageSpeed: {}'.format(averagePeakSpeed, peakSpeed, averageSpeed))

            for speed, acc in zip(linearSpeedList, linearAccList):
                fig, (ax1, ax2) = plt.subplots(1, 2) 
                speed = np.array(speed)
                print(speed.shape)
                ax1.plot(np.arange(len(speed)) , speed)
                ax1.set_ylabel('linear Speed Value')
                # if 'linearAccList' in resDict.keys():
                ax2.plot(np.arange(len(acc)) , acc)
                ax2.set_ylabel('linear acc Value')
                plt.show()

            # for speed, acc in zip(linearSpeedList, linearAccList):
            #     fig, (ax1, ax2) = plt.subplots(1, 2) 
            #     ax1.plot(np.arange(len(speed)) , speed)
            #     ax1.set_ylabel('linear Speed Value')
            #     ax2.plot(np.arange(len(acc)) , acc)
            #     ax2.set_ylabel('linear acc Value')
            #     plt.show()
        

def main():
    benchmarksResultsDir = '/home/majd/catkin_ws/src/basic_rl_agent/data/deep_learning/benchmarks/results'
    # configNumsList = ['config17', 'config62', 'config61']
    # configNumsList = ['config17']
    configNumsList = None #['config17']
    # configNumsList = ['config17_BeizerLoss_imageToBezierData1_1800_20210905-1315_benchmarkerPosesFile_#100_202109052231_28_frameMode1_202208121850_05.pkl']
    # configNumsList = ['config17_BeizerLoss_imageToBezierData1_1800_20210905-1315_benchmarkerPosesFile_#100_202109052231_28_frameMode1_202208141736_49.pkl']
    # configNumsList = ['config17_BeizerLoss_imageToBezierData1_1800_20210905-1315_benchmarkerPosesFile_#100_202205081959_38_frameMode1_202208141948_23.pkl']
    # configNumsList = ['rpg_sim2real_test_benchmark_benchmarkerPosesFile_#100_202205081959_38_frameMode1_202208142032_03.pkl']
    configNumsList = ['config17_BeizerLoss_imageToBezierData1_1800_20210905-1315_benchmarkerPosesFile_nonStationary_#50_20230101-124214_frameMode1_202301011459_18.pkl']
    benchmarkAnalyzer = BenchmarkAnalyzer(benchmarksResultsDir, configNumsList)

        

if __name__ == '__main__':
    main()