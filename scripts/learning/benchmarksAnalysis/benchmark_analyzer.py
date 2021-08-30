import os
import math
import numpy as np
import pandas as pd
import pickle

class BenchmarkAnalyzer:

    def __init__(self, benchmarksResultsDir):
        self.bechmarkResultsDir = benchmarksResultsDir
        self.ListOfDicts = []

        for file in os.listdir(benchmarksResultsDir)[:]:
            self.processBechmarkResultFile(file)
        
        sortedListOfDicts = sorted(self.ListOfDicts, key=lambda x: x['numOfSuccesses'], reverse=True)
        for dict in sortedListOfDicts:
            print(dict['fileName'], dict['numOfSuccesses'], dict['averagePeakSpeed'])
 

    def processBechmarkResultFile(self, file):
        resDict = pd.read_pickle(os.path.join(self.bechmarkResultsDir, file))

        numOfSuccesses = 0
        peakSpeeedList = []
        for finish_reason in resDict['round_finish_reason']:
            if finish_reason == 'dronePassedGate' or finish_reason == 'droneInFrontOfGate':
                numOfSuccesses += 1
                peakSpeeedList.append(resDict['peak_twist'])


        averagePeakSpeed = np.mean(peakSpeeedList) if len(peakSpeeedList) != 0 else math.nan

        resDict['numOfSuccesses'] = numOfSuccesses
        resDict['averagePeakSpeed'] = averagePeakSpeed
        resDict['fileName'] = file

        self.ListOfDicts.append(resDict)

def main():
    benchmarksResultsDir = '/home/majd/catkin_ws/src/basic_rl_agent/data/deep_learning/benchmarks/results'
    benchmarkAnalyzer = BenchmarkAnalyzer(benchmarksResultsDir)

        

if __name__ == '__main__':
    main()