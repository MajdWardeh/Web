from store_read_data import Data_Writer, Data_Reader, check_store_restore
from store_read_data_extended import DataWriterExtended, DataReaderExtended
import numpy as np
import pandas as pd

class DataWriterExtendedWithMarkers(DataWriterExtended):

    def __init__(self, file_name, dt, sample_length, max_samples, image_dimentions, velocity_shape):
        super(DataWriterExtendedWithMarkers, self).__init__(file_name, dt, sample_length, max_samples, image_dimentions, velocity_shape)

    def addSample(self, px, py, pz, yaw, imagesList, nsecsList, vel_data):
        canAddSample = super(DataWriterExtendedWithMarkers, self).addSample(px, py, pz, yaw, imagesList, nsecsList, vel_data)
        if canAddSample:
            # add data to the markers 
            raise NotImplementedError('not implemented!')
        return canAddSample
    
    
    def save_data(self):
        super(DataWriterExtendedWithMarkers, self).save_data()

      
    def __del__(self):
        print('destructor for DataWriterExtendedWithMarkers is called.')


class DataReaderExtended(Data_Reader):

    def __init__(self, file_name):
        super(DataReaderExtended, self).__init__(file_name)
        self.vel_df = pd.read_pickle('{}.pkl'.format(file_name))
        self.vel_list = self.vel_df['vel'].tolist() 
    
    def getVelSamples(self):
        return self.vel_list
    
    def getVelShape(self):
        return self.vel_list[0].shape



def test1_extended():
    dt = 0.0000000007
    sampleLength = 10
    numOfSamples = 10 
    numOfImageSequences = 3
    numOfImageChannels = 2
    vel_shape = (100000, 4)
    print('creating a test Data_Witer object...')
    dw = DataWriterExtended('test', dt, sampleLength, numOfSamples, (numOfImageSequences, numOfImageChannels), vel_shape)
    px_list_write = []
    py_list_write = []
    pz_list_write = []
    yaw_list_write = []
    nsecsList = range(numOfImageSequences-1)
    vel_data_list = []
    for i in range(numOfSamples):
        px = list(np.random.rand(sampleLength))
        py = list(np.random.rand(sampleLength))
        pz = list(np.random.rand(sampleLength))
        yaw = list(np.random.rand(sampleLength))
        px_list_write.append(px[:])
        py_list_write.append(py[:])
        pz_list_write.append(pz[:])
        yaw_list_write.append(yaw[:])
        nsecsList.append(i+numOfImageSequences-1) 
        l0 = [None]*numOfImageSequences
        image_list = [l0]*numOfImageChannels
        vel_data = np.random.rand(vel_shape[0], vel_shape[1])
        vel_data_list.append(vel_data)
        dw.addSample(px, py, pz, yaw, image_list, nsecsList[-numOfImageSequences:], vel_data)
    dw.save_data()
    print('creating a Data_Reader object...')
    dr = DataReaderExtended('test')
    indices, images, Px, Py, Pz, Yaw = dr.getSamples()
    vel_data_list_restored = dr.getVelSamples()

    print("check store/restore...")
    print("restored images names for the first sample is:")
    print(images[0])
    storeLists = [px_list_write, py_list_write, pz_list_write, yaw_list_write]
    restoreLists = [Px, Py, Pz, Yaw]
    comparsion_result = True
    for i in range(len(storeLists)):
        comparsion_result = comparsion_result and check_store_restore(storeLists[i], restoreLists[i])
    if comparsion_result == True:
        print("correct store/restore")
    else:
        print("the store/restore are not correct")
    print("checking velocity data...")
    if np.allclose(vel_data_list, vel_data_list_restored):
        print("velocity data restored correctly.")
    else:
        print("resotred velocity data is not close to the stored one.")
    keyList = list(dw.nameImageDictionary.keys())
    keyList.sort()
    print(keyList)


def main():
    test1_extended()


if __name__ == '__main__':
    main()