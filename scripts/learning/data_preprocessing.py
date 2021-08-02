import pandas as pd
import numpy as np

def preprocessAllData(directory):
    df = pd.read_pickle(directory)

    # process images: removing the list
    imagesList = df['images'].tolist()
    imagesList = [image[0][0] for image in imagesList]
    print(imagesList[0])
    df.drop('images', axis = 1, inplace = True)
    df['images'] = imagesList

    # process positionControlPoints: remove a0=(0, 0, 0) from the np arrays.
    pcps = df['positionControlPoints'].tolist()
    pcps = [p[1:] for p in pcps]
    df.drop('positionControlPoints', axis = 1, inplace = True)
    df['positionControlPoints'] = pcps

    return df


def main():
    directory = '/home/majd/catkin_ws/src/basic_rl_agent/data/testing_data/allData.pkl'
    df = preprocessAllData(directory)
    twistData = df['vel'].tolist()
    positionCP = df['positionControlPoints'].tolist()
    # print(twistData[0])
    # print(twistData[0][-10:, :])
    print(positionCP[0][:])
    cpx = np.zeros((4, ))
    for i, point in enumerate(positionCP[0]):
        cpx[i] = point[0]
    print(cpx)

    positionCP_nparray = np.array(positionCP[0])
    cpx = positionCP_nparray[:, 0]
    print(cpx)



if __name__ == '__main__':
    main()