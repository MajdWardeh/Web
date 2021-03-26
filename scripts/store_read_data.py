import numpy as np
from struct import pack, unpack
import cv2

class Data_Writer:
    # Manage the storing of the data collected from the path which is:
    # a number of sampled short-trajectories
    # 
    # The data is stored in a number of files:
    # 1- one txt file which contains:
    #    a. dt, sample_length, number_of_samples
    #    b. the number (index) and name of the images saved.
    # 2- Four files that contains the sampled short trajectories.
    #    Each file contains the index of a sample, followed by the sample on an axs.

    def __init__(self, file_name, dt, sample_length, max_samples, image_dimentions):
        #image_dimentions = (w, h) is a tuple describing the saved images.
        # w: describs the number of sequenced images to be stored for a given sample.
        # h: describs the number of images that represent the input of the network. 

        self.file_name = file_name
        self.dt = dt
        self.sample_length = sample_length
        assert max_samples > 0, "max_samples must be greater that zero"
        self.max_samples = max_samples
        self.CanAddSample = True 
        self.data_saved = False

        self.numOfSequencedImages, self.numOfInputImages = image_dimentions
        assert self.numOfInputImages > 0, "number of input images must be greater than zero"
        assert self.numOfSequencedImages > 0, "number of sequenced images must be greater than zero"
        self.imagesToSaveList = []

        self.txt_file = open('{}.txt'.format(self.file_name), 'w') # 'x' if the file exists, the operation files.
        self.px_file = open('{}X'.format(self.file_name), 'wb')
        self.py_file = open('{}Y'.format(self.file_name), 'wb')
        self.pz_file = open('{}Z'.format(self.file_name), 'wb')
        self.yaw_file = open('{}Yaw'.format(self.file_name), 'wb')

        self.txt_string = ''
        self.index = 0
        self.Px, self.Py, self.Pz, self.Yaw = [], [], [], []

    def addSample(self, px, py, pz, yaw, imagesList):
        #imagesList is a numpy array. imagesList.shape = (numOfSequencedImages, numOfInputImages)
        if self.CanAddSample: 
            assert imagesList.shape == (self.numOfSequencedImages, self.numOfInputImages), "The shape of imagesList is not correct"
            #process self.txt_string
            self.txt_string += '{}'.format(self.index)
            for i in range(self.numOfSequencedImages):
                for j in range(self.numOfInputImages):
                    image_name = '{}_im{}_{}{}.jpg'.format(self.file_name, self.index, i, j)
                    self.txt_string += ' {}'.format(image_name) 
                    self.imagesToSaveList.append(imagesList[i][j])
            self.txt_string += '\n'

            #check the sample length
            for l in [px, py, pz, yaw]:
                assert len(l) == self.sample_length, 'Error: added sample length ({}) does not match the sample_length ({})'.format(len(l), self.sample_length)
            px.insert(0, self.index)
            py.insert(0, self.index)
            pz.insert(0, self.index)
            yaw.insert(0, self.index)
            self.Px += px
            self.Py += py
            self.Pz += pz
            self.Yaw += yaw

            self.index += 1
            self.CanAddSample = self.index < self.max_samples
            return True
        else:
            return False

    def save_images(self):
        #to be edited
        for i, image in enumerate(imageList):
            image_name = '{}_im{}.jpg'.format(self.file_name, i)
            cv2.imwrite(image_name, image)

    def save_data(self):
        start_txt_file = '{} {} {} {} {}\n'.format(str(self.dt), self.sample_length, self.index, self.numOfSequencedImages, self.numOfInputImages)
        self.txt_file.write(start_txt_file + self.txt_string)
        array_len = len(self.Px)
        self.px_file.write(pack('d' * array_len, *self.Px))
        self.py_file.write(pack('d' * array_len, *self.Py))
        self.pz_file.write(pack('d' * array_len, *self.Pz))
        self.yaw_file.write(pack('d' * array_len, *self.Yaw))

        self.txt_file.close()
        self.px_file.close()
        self.py_file.close()
        self.pz_file.close()
        self.yaw_file.close()
        self.data_saved = True

class Data_Reader:

    def __init__(self, file_name):
        self.txt_file = open('{}.txt'.format(file_name), 'r')
        self.px_file = open('{}X'.format(file_name), 'rb')
        self.py_file = open('{}Y'.format(file_name), 'rb')
        self.pz_file = open('{}Z'.format(file_name), 'rb')
        self.yaw_file = open('{}Yaw'.format(file_name), 'rb')
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

#For debugging
def check_store_restore(list1, list2):
    list1, list2 = np.array(list1), np.array(list2)
    print(list1.shape, list2.shape)
    return (list1 == list2).all()

def main():
    dw = Data_Writer('test', 0.0000000007, 10, 10000, (3, 4)) 
    px_list_write = []
    py_list_write = []
    pz_list_write = []
    yaw_list_write = []
    for i in range(10000):
        px = list(np.random.rand(10))
        py = list(np.random.rand(10))
        pz = list(np.random.rand(10))
        yaw = list(np.random.rand(10))
        px_list_write.append(px[:])
        py_list_write.append(py[:])
        pz_list_write.append(pz[:])
        yaw_list_write.append(yaw[:])

        dw.addSample(px, py, pz, yaw, np.random.rand(3, 4))
    dw.save_data()
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
    
if __name__ == "__main__":
    main()