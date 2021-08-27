from genericpath import isfile
import os 
import numpy as np
from numpy.lib.function_base import average
import pandas as pd
import matplotlib.pyplot as plt

workingDir = '/home/majd/catkin_ws/src/basic_rl_agent/data/deep_learning/MarkersToBezierDataFolder/trainHistoryDict'

class HistoryAnalyzer:

    def __init__(self):
        self.file_statistics_dict = {}

    def processHistoryFile(self, file, verbose=True):
        histDict = pd.read_pickle(file)
        loss = np.array(histDict['loss'])
        val_loss = np.array(histDict['val_loss'])

        min_loss = np.min(loss)
        min_val_loss = np.min(val_loss)

        average_loss = np.mean(loss[-100:])
        average_val_loss = np.mean(val_loss[-100:])
         
        if verbose and min_loss < 0.1:
            print('File: {}'.format(file.split('/')[-1]))
            print('min loss: {}, average loss: {}'.format(min_loss, average_loss))
            print('min val_loss: {}, average val_loss: {}'.format(min_val_loss, average_val_loss))
            print('-----------------------------------------')
        
        return [min_loss, average_loss, min_val_loss, average_val_loss]
    
    def findBestHistoryFile(self, files):
        averageValLoss_file_dict = {}
        min_average_val_loss = 100000
        min_file = None
        for file in files:
            returnedList = self.processHistoryFile(file, verbose=False)
            average_val_loss = returnedList[-1]
            averageValLoss_file_dict[average_val_loss] = file
            if average_val_loss < min_average_val_loss:
                min_average_val_loss = average_val_loss
                min_file = file 
        print('minFile: {}, min_average_val_loss: {}'.format(min_file.split('/')[-1], min_average_val_loss)) 


def main():

    targetConfigNums = [15, 16, 17, 18, 19, 20]

    historyFiles = [os.path.join(workingDir, file) for file in os.listdir(workingDir) if file.endswith('.pkl')]

    targetFiles = []
    for file in historyFiles:
        for configNum in targetConfigNums:
            if 'config{}'.format(configNum) in file:
                targetFiles.append(file)
                break

    historyAnalyzer = HistoryAnalyzer()

    # historyAnalyzer.findBestHistoryFile(targetFiles)

    for file in targetFiles:
        historyAnalyzer.processHistoryFile(file)




if __name__ == '__main__':
    main()