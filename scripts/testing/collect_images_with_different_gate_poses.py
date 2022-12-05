import os
import subprocess
import time

from markerDataCollector import ImageMarkersDataCollector

def writeToFile(gatesLocations, dir):
    path = os.path.join(dir, 'gateLocationsFile.txt')
    gatesLocationsDict = {}
    with open(path, 'w+') as f:
        for i, gate in enumerate(gatesLocations):
            gatesLocationsDict['gate{}B'.format(i)] = gate
            f.write('gate{}B: {}, {}, {}, {}, 2, 2, 2\n'.format(i, gate[0], gate[1], gate[2], gate[3]))
    return gatesLocationsDict

def main():
    base_dir = '/home/majd/catkin_ws/src/basic_rl_agent/scripts/testing'
    markersLocationsDir = base_dir
    epochs = 1 #10
    samplesNum = 2 #1000
    gate_location_list = [
                [0, 0, 2, 90],
                # [3, 0, 3, 80],
                # [-4, 6, 4, 90],
                # [0, 0, 5, 90],
                # [3, -3, 4, -70],
                # [0, -6, 5, 90],
                # [0, 0, 2, 90],
                # [0, 0, 2, -70],
                # [0, 0, 2, 60],
                # [0, 0, 2, 120],
    ]

    imageMarkersCollector = ImageMarkersDataCollector(markersLocationsDir)
    for loc in gate_location_list[:]: 
        writeToFile([loc], base_dir)
        time.sleep(0.5)
        flightGogglesProcess = subprocess.Popen(['/home/majd/catkin_ws/src/basic_rl_agent/data/FG_linux/FG_gatesPlacementFileV2/FG_gatesPlacementFileV2.x86_64'],
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE)
        time.sleep(13)
        imageMarkersCollector.updateMarkersLocation()
        imageMarkersCollector.run(samplesNum, epochs)

        # markersDataCollectorProcess.wait()
        # markersDataCollectorProcess.kill()
        flightGogglesProcess.kill()


if __name__ == '__main__':
    main()




    

