import pandas as pd

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
    print(twistData[0])
    print(len(twistData))    

if __name__ == '__main__':
    main()