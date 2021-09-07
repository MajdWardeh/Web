import os
import math
import numpy as np
import pandas as pd
import pickle

class BenchmarkAnalyzer:

    def __init__(self, benchmarksResultsDir, configNumsList=None):
        self.bechmarkResultsDir = benchmarksResultsDir
        self.ListOfDicts = []

        if configNumsList is None:
            for file in os.listdir(benchmarksResultsDir)[:]:
                self.processBechmarkResultFile(file)
        else:
            for file in os.listdir(benchmarksResultsDir)[:]:
                for configNum in configNumsList:
                    if configNum in file:
                        self.processBechmarkResultFile(file)
        
        sortedListOfDicts = sorted(self.ListOfDicts, key=lambda x: x['numOfSuccesses'], reverse=True)
        for dict in sortedListOfDicts:
            print(dict['fileName'], dict['numOfSuccesses'], dict['averagePeakSpeed'], dict['avergeSpeed'])
 

    def processBechmarkResultFile(self, file):
        resDict = pd.read_pickle(os.path.join(self.bechmarkResultsDir, file))

        numOfSuccesses = 0
        peakSpeeedList = []
        avergeSpeedList = []
        for finish_reason in resDict['round_finish_reason']:
            if finish_reason == 'dronePassedGate' or finish_reason == 'droneInFrontOfGate':
                numOfSuccesses += 1
                peakSpeeedList.append(resDict['peak_twist'])
                avergeSpeedList.append(resDict['average_twist'])


        averagePeakSpeed = np.mean(peakSpeeedList) if len(peakSpeeedList) != 0 else math.nan
        averageSpeed = np.mean(avergeSpeedList) if len(avergeSpeedList) != 0 else math.nan

        resDict['numOfSuccesses'] = numOfSuccesses
        resDict['averagePeakSpeed'] = averagePeakSpeed
        resDict['avergeSpeed'] = averageSpeed
        resDict['fileName'] = file

        self.ListOfDicts.append(resDict)

def main():
    benchmarksResultsDir = '/home/majd/catkin_ws/src/basic_rl_agent/data/deep_learning/benchmarks/results'
    # configNumsList = ['config17', 'config62', 'config61']
    configNumsList = None
    benchmarkAnalyzer = BenchmarkAnalyzer(benchmarksResultsDir, configNumsList)

        

if __name__ == '__main__':
    main()