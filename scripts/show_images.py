import cv2 
import os


SAVE_DATA_DIR = '/home/majd/drone_racing_ws/catkin_ddr/src/basic_rl_agent/data/testing_data'
dataset_files = os.listdir(SAVE_DATA_DIR)
for file in dataset_files:
    if file.endswith('.jpg'):
        im = cv2.imread(os.path.join(SAVE_DATA_DIR, file))
        cv2.imshow('image', im)
        cv2.waitKey(0)
cv2.closeAllWindows()

