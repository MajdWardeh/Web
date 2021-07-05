import cv2 
import os


SAVE_DATA_DIR = '/home/majd/catkin_ws/src/basic_rl_agent/data/testing_data' #'/home/majd/drone_racing_ws/catkin_ddr/src/basic_rl_agent/data/testing_data'
dataset_files = os.listdir(SAVE_DATA_DIR)
for folder in dataset_files:
    for file in os.listdir(os.path.join(SAVE_DATA_DIR, folder)):
        if file.endswith('.jpg'):
            im = cv2.imread(os.path.join(SAVE_DATA_DIR, folder, file))
            cv2.imshow('image', im)
            cv2.waitKey(3)
# cv2.closeAllWindows()

