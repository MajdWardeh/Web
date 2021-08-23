import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2
sys.path.append(ros_path)


import os
import time
import numpy as np
from struct import pack, unpack

import pandas as pd

class Data_Writer(object):
    # Manage the storing of the data collected from the path which is:
    # a number of sampled short-trajectories
    # 
    # The data is stored in a number of files:
    # 1- one txt file which contains:
    #    a. dt, sample_length, number_of_samples
    #    b. the number (_index) and name of the images saved.
    # 2- Four files that contains the sampled short trajectories.
    #    Each file contains the _index of a sample, followed by the sample on an axs.

    def __init__(self, dataset_path, dt, sample_length, max_samples, image_dimentions, storeMarkers=False):
        '''
            @param image_dimentions = (w, h) is a tuple describing the saved images.
                w: describs the number of sequenced images to be stored for a given sample.
                h: describs the number of images (channels e.g 1 RGB and 1 depth corresponds to h=2) that represent the input of the network. 
            @param storeMarkers: boolean default=False. If True, markersData will be saved and it must be proveded in the function 'AddSample'.
        
        '''

        assert os.path.exists(dataset_path), 'the provided dataset_path does not exit {}'.format(dataset_path)
        self.imagesPath = os.path.join(dataset_path, 'images')
        self.dataPath = os.path.join(dataset_path, 'data')
        for path in [self.imagesPath, self.dataPath]:
            if not os.path.exists(path):
                os.mkdir(path)
        randomName = 'data_{:d}'.format(int(round(time.time() * 1000)))
        self.file_name_data = os.path.join(self.dataPath, randomName)
        self.file_name_images = os.path.join(self.imagesPath, randomName)


        self.dt = dt
        self.sample_length = sample_length
        assert max_samples > 0, "max_samples must be greater that zero"
        self.max_samples = max_samples
        self._canAddSample = True 
        self.data_saved = False

        self.numOfSequencedImages, self.numOfInputImages = image_dimentions
        assert self.numOfInputImages > 0, "number of input images must be greater than zero"
        assert self.numOfSequencedImages > 0, "number of sequenced images must be greater than zero"
        self.imageId_set = set()
        self.nameImageDictionary = {}

        self.txt_string = ''
        self._index = 0
        self.Px, self.Py, self.Pz, self.Yaw = [], [], [], []

        self.storeMarkers = storeMarkers
        if self.storeMarkers:
            self.imageMarkersDataDict = {}
            self.markersDataExpectedShape = (4, 3)

    def addSample(self, px, py, pz, yaw, imagesList, nsecsList, markersDataList=None):
        '''
            @param imagesList is a list of lists of images. len(imagesList) = numOfSequencedImages, len(imageList[0]) = numOfInputImages.
            @param nsecsList is a list storing the nanoseconds of the time stamps of the messages. It is used as an ID for the images in order to know if it is 
                    stored or not (in order to not storing an image mutiple times).
            @param markersDataList is a list of numpy arrays. len(markersDataList) = numOfSequencedImages.
                    Each np array 'markersData' has shape=(4, 3), the order of markers in markersData refers to the index
                    of each marker in the image.
        '''
        if self._canAddSample: 
            # adding the data to the variables.
            self._addSample(px, py, pz, yaw, imagesList, nsecsList, markersDataList)
            # updating the _index and checking if we can add other samples.
            self._index += 1
            self._canAddSample = self._index < self.max_samples
            return True
        else:
            return False

    def _addSample(self, px, py, pz, yaw, imagesList, nsecsList, markersDataList):
            dims_found = (len(imagesList[0]), len(imagesList))
            assert dims_found == (self.numOfSequencedImages, self.numOfInputImages), "The shape of imagesList is not correct, expected: {}, found: {}".format((self.numOfSequencedImages, self.numOfInputImages), dims_found)
            #process self.txt_string:
            self.txt_string += '{}'.format(self._index)
            for i in range(self.numOfSequencedImages):
                for j in range(self.numOfInputImages):
                    imageId = '{}_{}'.format(j, nsecsList[i])
                    #Does the image id already exit?
                    image_name = '{}_im{}.jpg'.format(self.file_name_images, imageId)
                    self.txt_string += ' {}'.format(image_name) 
                    if imageId not in self.imageId_set:
                        self.imageId_set.add(imageId)
                        self.nameImageDictionary[image_name] = imagesList[j][i] 
                        if self.storeMarkers and j==0: # index j=0 in self.numOfInputImages points to the RGB images
                            assert not markersDataList is None, 'markersDataList is either None or not provided.' 
                            markersData = markersDataList[i] 
                            assert markersData.shape == self.markersDataExpectedShape, 'the shape of markersData in markersDataList is not as expected, expected={}, found={}'.format(self.markersDataExpectedShape, markersData.shape)
                            self.imageMarkersDataDict[image_name] = markersData
            self.txt_string += '\n'
            #check the sample length
            for l in [px, py, pz, yaw]:
                assert len(l) == self.sample_length, 'Error: added sample length ({}) does not match the sample_length ({})'.format(len(l), self.sample_length)
            px.insert(0, self._index)
            py.insert(0, self._index)
            pz.insert(0, self._index)
            yaw.insert(0, self._index)
            self.Px += px
            self.Py += py
            self.Pz += pz
            self.Yaw += yaw

    def CanAddSample(self):
        return self._canAddSample
       
    def getIndex(self):
        return self._index
        
    def save_images(self):
        for image_name in self.nameImageDictionary:
            cv2.imwrite(image_name, self.nameImageDictionary[image_name])

    # debugging markersData
    # def save_images(self):
    #     for image_name in self.nameImageDictionary:
    #         assert image_name in self.imageMarkersDataDict, '{} is not in imageMarkersDataDict'
    #         markersData = self.imageMarkersDataDict[image_name]
    #         image = self.nameImageDictionary[image_name]
    #         for marker in markersData:
    #             marker = marker.astype(np.int)
    #             image = cv2.circle(image, (marker[0], marker[1]), radius=3, color=(0, 0, 255), thickness=-1)
    #         cv2.imwrite(image_name, image)

    def save_data(self):
        if self.data_saved == True:
            return
        start_txt_file = self._process_txt_header()
        self._save_data(start_txt_file) 
        self.save_images()
        time.sleep(3)
        self.data_saved = True

    def _save_data(self, start_txt_file):
        self.txt_file = open('{}.txt'.format(self.file_name_data), 'w') # 'x' if the file exists, the operation files.
        self.px_file = open('{}.X'.format(self.file_name_data), 'wb')
        self.py_file = open('{}.Y'.format(self.file_name_data), 'wb')
        self.pz_file = open('{}.Z'.format(self.file_name_data), 'wb')
        self.yaw_file = open('{}.Yaw'.format(self.file_name_data), 'wb')
        # add stored data to the files
        self.txt_file.write(start_txt_file + self.txt_string)
        array_len = len(self.Px)
        self.px_file.write(pack('d' * array_len, *self.Px))
        self.py_file.write(pack('d' * array_len, *self.Py))
        self.pz_file.write(pack('d' * array_len, *self.Pz))
        self.yaw_file.write(pack('d' * array_len, *self.Yaw))
        # close all the files
        self.txt_file.close()
        self.px_file.close()
        self.py_file.close()
        self.pz_file.close()
        self.yaw_file.close()
        # store markersData if provided
        if self.storeMarkers:
            imageNameList = []
            markersDataList = []
            for key, value in self.imageMarkersDataDict.items():
                imageNameList.append(key)
                markersDataList.append(value)
            # imagesList = self.imageMarkersDataDict.keys()
            # markersDataList = [self.imageMarkersDataDict[image] for image in imagesList]
            df = pd.DataFrame({
                'images': imageNameList,
                'markersData': markersDataList
            })
            df.to_pickle('{}_markersData.pkl'.format(self.file_name_data))
    
    def _process_txt_header(self):
        return '{} {} {} {} {}\n'.format(str(self.dt), self.sample_length, self._index, self.numOfSequencedImages, self.numOfInputImages)

class Data_Reader(object):

    def __init__(self, file_name):
        # raise NotImplementedError('changed file_name to 2 files: file_name_data, file_name_images')
        self.txt_file = open('{}.txt'.format(file_name), 'r')
        self.px_file = open('{}.X'.format(file_name), 'rb')
        self.py_file = open('{}.Y'.format(file_name), 'rb')
        self.pz_file = open('{}.Z'.format(file_name), 'rb')
        self.yaw_file = open('{}.Yaw'.format(file_name), 'rb')
        txt_lines = self.txt_file.readlines()
        self.__process_txt_file(txt_lines)

    def __process_txt_file(self, txt_lines):
        first_line = txt_lines[0][:-1]
        dt, sample_length, number_of_samples, numOfSequencedImages, numOfInputImages = first_line.split(' ')
        self.dt = float(dt)
        self.sample_length = int(sample_length)
        self.number_of_samples = int(number_of_samples)
        self.numOfSequencedImages = int(numOfSequencedImages)
        self.numOfInputImages = int(numOfInputImages)
        self.image_indices = []
        self.images = []
        for line in txt_lines[1:]:
            strings = line[:-1].split(' ')
            self.image_indices.append(int(strings[0]))
            strings = strings[1:]
            images = []
            for i in range(self.numOfSequencedImages):
                image = []
                for j in range(self.numOfInputImages):
                   image.append(strings[i*self.numOfInputImages+j]) 
                images.append(image)
            self.images.append(np.vstack(images))
            
    def getSamples(self):
        Px_indices, self.Px_data = self.__process_data_file(self.px_file)
        Py_indices, self.Py_data = self.__process_data_file(self.py_file)
        Pz_indices, self.Pz_data = self.__process_data_file(self.pz_file)
        yaw_indices, self.yaw_data = self.__process_data_file(self.yaw_file)

        #check if the data restored correctly:
        Px_indices = np.array(Px_indices)
        Py_indices = np.array(Py_indices)
        Pz_indices = np.array(Pz_indices)
        yaw_indices = np.array(yaw_indices)
        check = np.array_equal(Px_indices, Py_indices) and np.array_equal(Px_indices, Pz_indices) and np.array_equal(Px_indices, yaw_indices)
        assert check == True, 'Error in restoring data, the data indices are not equal.'
        return self.image_indices, self.images, self.Px_data, self.Py_data, self.Pz_data, self.yaw_data

    def __process_data_file(self, file):
        packed = file.read()
        array = unpack('d' * (len(packed) // 8), packed) # 8 bytes per double
        array = list(array)
        indices_list = []
        data_list = []
        for i in range(0, self.number_of_samples*(self.sample_length + 1), self.sample_length+1):
            d = array[i:i+self.sample_length+1]
            indices_list.append(d[0])
            data_list.append(d[1:])
        return indices_list, data_list 
    
    def getDt(self):
        return self.dt
    
    def getNumOfSamples(self):
        return self.number_of_samples
    
    def getNumOfImageSequence(self):
        return self.numOfSequencedImages

#For debugging
def check_store_restore(list1, list2):
    list1, list2 = np.array(list1), np.array(list2)
    print(list1.shape, list2.shape)
    return (list1 == list2).all()

def test1():
    dt = 0.0000000007
    sampleLength = 10
    numOfSamples = 10 
    numOfImageSequences = 3
    numOfImageChannels = 2

    print('creating a test Data_Witer object...')
    dw = Data_Writer('test', dt, sampleLength, numOfSamples, (numOfImageSequences, numOfImageChannels)) 
    px_list_write = []
    py_list_write = []
    pz_list_write = []
    yaw_list_write = []
    nsecsList = range(numOfImageSequences-1)
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
        dw.addSample(px, py, pz, yaw, image_list, nsecsList[-numOfImageSequences:])
    dw.save_data()
    print('creating a Data_Reader object...')
    dr = Data_Reader('test')
    indices, images, Px, Py, Pz, Yaw = dr.getSamples()

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

    keyList = list(dw.nameImageDictionary.keys())
    keyList.sort()
    print(keyList)


def main():
    test1()

if __name__ == "__main__":
    main()