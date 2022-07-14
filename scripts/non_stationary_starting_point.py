import os
import numpy as np
import numpy.linalg as la
import pandas as pd




def main():
    # np.random.seed(0)

    benchmarkPosesRootDir = '/home/majd/catkin_ws/src/basic_rl_agent/data/deep_learning/benchmarks/benchmarkPosesFiles'
    fileName = 'benchmarkerPosesFile_#5_202205211439_14.pkl'
    posesDataFrame = pd.read_pickle(os.path.join(benchmarkPosesRootDir, fileName))
    poses = posesDataFrame['poses'].tolist()

    max_vel = np.array([2, 2, 0.5, 0])
    d_xy = 4
    d_z = 1
    
    pe = poses[0]
    print(pe)
    ve = np.random.rand(4) * max_vel 
    ve_r_xy = la.norm(ve[:2])
    ve_theta_xy = np.arctan2(ve[1], ve[0])
    ve_theta_z = np.arctan(ve_r_xy/ve[2])

    print(ve)
    print(ve_r_xy)
    print(np.rad2deg(ve_theta_xy), np.rad2deg(ve_theta_z))

    xs = pe[0] - d_xy * np.cos(ve_theta_xy)
    ys = pe[1] - d_xy * np.sin(ve_theta_xy)
    zs = pe[2] - d_z * np.cos(ve_theta_z)
    phi_s = pe[3]

    print('pe: ', pe)
    print('ps: ', xs, ys, zs, phi_s)


if __name__ == '__main__':
    main()