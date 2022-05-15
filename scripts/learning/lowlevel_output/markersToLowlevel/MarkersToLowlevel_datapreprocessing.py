import os
import numpy as np
import pandas as pd
import yaml

def process_data(df, stds_dir):
    command_seq = np.array(df['commandSequence'].tolist())
    print(command_seq.shape)
    shape = (command_seq.shape[1], command_seq.shape[0] * command_seq.shape[2])
    command_seq_reshaped = np.zeros(shape)
    print(command_seq_reshaped.shape)
    for i in range(command_seq_reshaped.shape[0]):
        command_seq_reshaped[i] = command_seq[:, i, :].reshape(-1,)
    stds = [command_seq[:, i, :].std() for i in range(4)]
    print(stds)
    print(command_seq_reshaped.std(axis=1))
    command_stds = command_seq_reshaped.std(axis=1)
    new_commads_seq = (command_seq_reshaped.T / command_stds).T
    print(new_commads_seq.std(axis=1).shape)
    assert np.allclose(new_commads_seq.std(axis=1), np.ones((4, )) )
    stds_dict = {}
    for i in range(4):
        stds_dict[i] = float(command_stds[i])
    with open(os.path.join(stds_dir, 'stds_file.yaml'), 'w') as outfile:
        yaml.dump(stds_dict, outfile, default_flow_style=False)
    
def normalize_controlCommands(command_seq_list, stds_file_path):
    with open(stds_file_path, 'r') as stream:
        stds_dict = yaml.safe_load(stream)

    command_stds = np.array([stds_dict[i] for i in range(4)]) 

    command_seq = np.array(command_seq_list)
    command_seq_normalized = (1./command_stds).reshape(-1, 1) * command_seq
    command_name_list = ['thrust', 'roll', 'ptich', 'yaw']
    for i in range(4):
        command_i_std = command_seq_normalized[:, i, :].reshape(-1,).std()
        print('command: {} std: {}'.format(command_name_list[i], command_i_std))
        # assert np.allclose(command_i_std, 1.), 'assertion failed'
    return command_seq_normalized
    

def main():
    df_path = '/home/majd/catkin_ws/src/basic_rl_agent/data2/flightgoggles/datasets/imageLowLevelControl_1000/allData_imageLowLevelControl_1000_rowsCount246328_20220420-1328.pkl'
    stds_dir = '/home/majd/catkin_ws/src/basic_rl_agent/scripts/learning/lowlevel_output/markersToLowlevel' 


    df = pd.read_pickle(df_path)
    # process_data(df, stds_dir)
    command_seq_list = df['commandSequence'].tolist()
    stds_file_path = os.path.join(stds_dir, 'stds_file.yaml')
    normalize_controlCommands(command_seq_list, stds_file_path)


if __name__ == '__main__':
    main()