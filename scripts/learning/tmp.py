import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
import numpy as np

def main():
    x = np.random.randint(1, 10, (4, 3))
    y = np.array([0.2, 0.5, 1])
    xy = np.multiply(x, y)
    print(x)
    print(y)
    print(xy)



if __name__ == '__main__':
    main()