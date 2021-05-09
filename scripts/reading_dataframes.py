import os
import pandas as pd

workingDirectory = '.'
pklFileList = [file for file in os.listdir(workingDirectory) if file.endswith('.pkl')]
for pklFile in pklFileList:
    print('opening {}'.format(pklFile))
    df = pd.read_pickle(pklFile)
    print(df)