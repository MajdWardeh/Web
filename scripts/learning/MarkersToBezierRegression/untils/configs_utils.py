import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import math
import pandas as pd
import yaml


def configsToFile():
    configs = defineConfigs()
    configsDict = {}
    for config in configs:
        configsDict['config{}'.format(config['configNum'])] = config
    with open('./configs/configs_{}.yaml'.format(int(time.time())), 'w') as outfile:
        yaml.dump(configsDict, outfile, default_flow_style=False) 

def loadConfigsFromFile(yaml_file):
    with open(yaml_file, 'r') as stream:
        try:
            loadedConfigs = yaml.load(stream, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)
    return loadedConfigs

def loadConfigFiles(configDir):
    configFiles = []
    found_config_yaml_files = [file for file in os.listdir(configDir) if file.endswith('.yaml')]
    for configFile in found_config_yaml_files:
        configFiles.append(os.path.join(configDir, configFile))
    return configFiles

def loadAllConfigs(configRootDir, listOfConfigNums=None):
    '''
        if listOfConfigNums is None, it will return all the found configs.
    '''
    configFiles = loadConfigFiles(configRootDir) 
    allConfigs = {}
    for file in configFiles:
        configs = loadConfigsFromFile(file)
        # update the missing configs
        for config in configs.keys():
           configs[config]['numOfImageSequence'] = configs[config].get('numOfImageSequence', 1)
           configs[config]['markersNetworkType'] = configs[config].get('markersNetworkType', 'Dense')
           configs[config]['twistNetworkType'] = configs[config].get('twistNetworkType', 'Dense')
           configs[config]['twistDataGenType'] = configs[config].get('twistDataGenType', 'last2points')
        if listOfConfigNums is None:
            allConfigs.update(configs)
        else:
            tmpConfigs = {}
            for key, config in configs.items():
                if 'config{}'.format(config['configNum']) in listOfConfigNums:
                    tmpConfigs[key] = config
            if tmpConfigs:
                allConfigs.update(tmpConfigs)
    return allConfigs



def test():
    configsRootDir = '/home/majd/catkin_ws/src/basic_rl_agent/scripts/learning/MarkersToBezierRegression/configs'
    configsNumList = ['config15', 'config16']
    allConfigs = loadAllConfigs(configsRootDir, configsNumList)
    print(allConfigs)



if __name__ == '__main__':
    test()

