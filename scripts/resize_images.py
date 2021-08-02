import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2 
import time
import os


SAVE_DATA_DIR = '/home/majd/catkin_ws/src/basic_rl_agent/data/testing_data' #'/home/majd/drone_racing_ws/catkin_ddr/src/basic_rl_agent/data/testing_data'
dataset_files = os.listdir(SAVE_DATA_DIR)
for folder in dataset_files:
    if os.path.isdir(os.path.join(SAVE_DATA_DIR, folder)) == False:
        continue
    for file in os.listdir(os.path.join(SAVE_DATA_DIR, folder)):
        if file.endswith('.jpg'):
            image_name = os.path.join(SAVE_DATA_DIR, folder, file)
            im = cv2.imread(image_name)
            print('processing {}'.format(image_name), end=' ')
            if im is None:
                print('bad image.')
                os.remove(image_name)
                continue 
            if im.shape == (240, 320, 3):
                print('skipped.')
            else:
                im = cv2.resize(im, (320, 240))
                cv2.imwrite(image_name, im)
                time.sleep(0.002)
                print('resized.')
            # cv2.imshow('image', im)
            # cv2.waitKey(3)

# cv2.closeAllWindows()
print('done')

