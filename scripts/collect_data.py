from __future__ import print_function
import os
import signal
import numpy as np
import random
import time
import subprocess
import shutil
from imutils import paths
from data_generator import Dataset_collector, placeAndSimulate

# TODO:
# test changing the gate illumination.
# randomize the backgrounds. [done]
# write a log file
# kill all ros nodes when exit

# TRAIN_DIR="../../../../learning/deep_drone_racing_learner/data/Training"
RESOURCES_DIR = '/home/majd/drone_racing_ws/catkin_ddr/src/basic_rl_agent/resources'

def collect_data_in_fixed_env(num_iterations):
    for epoch in range(1):
        print("-----------------------------------------------")
        print("Epoch: #{}".format(epoch))
        print("-----------------------------------------------")
        collector = Dataset_collector()
        for i in range(5):
            placeAndSimulate(collector)
            if collector.maxSamplesAchived:
                break
    print("done.")


def main():
    bkg_folder=os.path.join(RESOURCES_DIR, 'race_track/iros_materials/materials/textures/train_bkgs')
    texture_goal_fname=os.path.join(RESOURCES_DIR, 'race_track/iros_materials/materials/textures/sky.jpg')
    asphalt_goal_fname=os.path.join(RESOURCES_DIR, 'race_track/iros_materials/materials/textures/asphalt.jpg')
    gate_material_folder=os.path.join(RESOURCES_DIR, 'race_track/iros_materials/materials/textures/gate_bkgs')
    bkg_goal_fname=os.path.join(RESOURCES_DIR, 'race_track/real_world/gate/meshes/images.jpeg')

    # Gate shapes
    gates_shapes_dir=os.path.join(RESOURCES_DIR, 'race_track/real_world/gate/meshes/gate_shapes')
    all_shapes = [os.path.join(gates_shapes_dir, f.split('.')[0]) \
                  for f in os.listdir(gates_shapes_dir) if f.endswith('.stl')]
    num_gates = len(all_shapes) - 1 # Last is used for testing
    gate_dae=os.path.join(RESOURCES_DIR, 'race_track/real_world/gate/meshes/gate.dae')
    gate_stl=os.path.join(RESOURCES_DIR, 'race_track/real_world/gate/meshes/gate.stl')
    light_changer=os.path.join(RESOURCES_DIR, 'race_track/real_world/gate/meshes/set_gate_properties.py')

    num_iterations_per_bkg = 1
    num_loops = 100 

    # Read all the backgrounds and order them
    all_images = paths.list_images(bkg_folder)
    all_images = sorted(all_images)

    # Read all possible gate materials
    all_gates_materials = paths.list_images(gate_material_folder)
    all_gates_materials = sorted(all_gates_materials)

    # if not os.path.isdir(TRAIN_DIR):
        # os.mkdir(TRAIN_DIR)

    all_images_with_index = zip(range(len(all_images)), all_images)
    random.shuffle(all_images_with_index)
    for loop_i in range(num_loops):
        print("################# {} #################".format(loop_i))
        background_round = 1
        for i, bkg_img_fname in all_images_with_index: #enumerate(all_images):
            # Copy new background
            os.system("cp {} {}".format(bkg_img_fname, texture_goal_fname))
            # Copy new asphalt
            os.system("cp {} {}".format(all_images[-(i+1)], asphalt_goal_fname))
            # Copy new gate background
            os.system("cp {} {}".format(all_gates_materials[i%9], bkg_goal_fname)) # Use the first 9 for training and the last for testing
            # Copy new gate shape
            gate_number = np.random.choice(num_gates)
            shutil.copy(all_shapes[gate_number]+ '.stl', gate_stl)
            shutil.copy(all_shapes[gate_number]+ '.dae', gate_dae)
            # Make random illumination
            os.system("python {} -xml_file {} -emission {} -ambient {}".format(
                light_changer,
                gate_dae,
                0.1*np.random.rand(), # Gates have little emission, 0.3
                np.random.rand())) # 0.5

            print("Processing Background {}".format(background_round))
            background_round += 1
            time.sleep(1)
            # set environment
            subprocess.call("roslaunch basic_rl_agent drone_and_controller.launch &", shell=True)
            time.sleep(10)
            collect_data_in_fixed_env(num_iterations_per_bkg)
            os.system("pkill -9 rviz; pkill -9 gzserver")
            # os.system("pkill -9 gzserver")
            time.sleep(1)


def signal_handler(sig, frame):
    os.system("pkill -9 rviz; pkill -9 gzserver")
    sys.exit(0)   

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    main()
