""" 
Basic data preprocessor for INTERACTION dataset.
Each node has its own coordinate system.

data.x = [num_agent, agent_features]
data.y = [num_agent, agent_future_trajectories]
data.edge_index = [num_edges, 2] 
data.edge_attr = [num_edges, edge_attribute]
data.map = [map_width, map_length]
data.target_mask = [num_nodes, 1] 

edge_index will include:
    1. agent-agent edges
        1. relative position
        2. relative yaw angle
    2. map-agent edges
        1. agents position in the map
        2. agents yaw angle in the map

target_mask contains the boolean indicating whether the node is to be predicted. 
a node could be an agent or a map.
"""

import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import argparse

import math
import numpy as np
import torch
from PIL import Image, ImageOps

from torch_geometric.data import Data
from utils import map_vis_xy

class IT_DATA_PRE():
    def __init__(self, csv_names, save_path, root, stage, seg_len=40, gap_len=1, hist_len=10, fut_len=30, map_pad_width=120):
        self.csv_names = csv_names
        self.save_path = save_path
        self.root_path = root
        self.stage = stage
        self.seg_len = seg_len
        self.gap_len = gap_len
        self.hist_len = hist_len
        self.fut_len = fut_len
        self.pad_pxl_width = map_pad_width
        self.pxl_lengths = {'x': 0.25, 'y': 0.25}
        self.node_type_to_indicator_vec = { 'car': torch.tensor([[0,0,1]]),
                                            'pedestrian/bicycle': torch.tensor([[0,1,0]]),
                                            'map': torch.tensor([[1,0,0]])}
    

    def plot_map(self):
        fig, axes = plt.subplots(1, 1)
        fig.canvas.set_window_title("Interaction Dataset Visualization")
        lat_origin, lon_origin = 0. , 0.  # origin is necessary to correctly project the lat lon values in the osm file to the local
        map_vis_xy.draw_map_without_lanelet_xy(self.cur_map_path, axes, lat_origin, lon_origin)

        x_ax_limits, y_ax_limits = axes.get_xlim(), axes.get_ylim()
        # plt.show()
        plt.close()
        map_center_x = 0.5 * (x_ax_limits[1] + x_ax_limits[0])
        map_center_y = 0.5 * (y_ax_limits[1] + y_ax_limits[0])
        return {'map_xlim': x_ax_limits, 'map_ylim': y_ax_limits, 'map_center': [map_center_x, map_center_y]}

    def Srotate(self, angle, valuex, valuey, pointx, pointy):
        sRotatex = (valuex-pointx)*math.cos(angle) + (valuey-pointy)*math.sin(angle) + pointx
        sRotatey = (valuey-pointy)*math.cos(angle) - (valuex-pointx)*math.sin(angle) + pointy
        return sRotatex, sRotatey


    def generate_pyg(self, dataframe, scene, case):
        """ given a frame, preprocess the frame to get a pyg data for this frame """
        ### Processing agents including car and pedestrian
        agent_ids = set(dataframe['track_id'].values)
        agent_ids_drop = []
        for agent_id in agent_ids:
            current_exists = dataframe[(dataframe['track_id'] == agent_id) & (dataframe['frame_id'] == 10) ].values.shape[0]
            if current_exists:
                agent_ids_drop.append(agent_id)
        num_agent = len(agent_ids_drop)
        node_agent_idmap = {}
        for node_id, agent_id in enumerate(agent_ids_drop):
            node_agent_idmap[agent_id] = node_id

        ## Node features
        node_feats = torch.zeros((len(agent_ids_drop), self.hist_len, 4)) 
        gt = torch.zeros((len(agent_ids_drop), self.fut_len, 2)) 

        ## Edge
        edge_feats = torch.empty((2, 0)).long()
        edge_attrs = torch.empty((5, 0))
        edge_types = torch.empty((6, 0)).long()

        ## Vehicle to map attributes
        map_attrs = torch.empty((5, 0))

        ## Masks
        target_mask = []
        car_target_mask = []

        car_mask = []
        ped_mask = []
  
        map_ctr = torch.tensor(self.map_info['map_center'])

        ## Raw hist and fut
        hist_raw = torch.zeros((len(agent_ids_drop), self.hist_len, 4))
        future_raw = torch.zeros((len(agent_ids_drop), self.fut_len, 2))

        for i, agent_id in enumerate(agent_ids_drop):

            agent_df = dataframe[dataframe['track_id'] == agent_id]
            agent_pos = agent_df[agent_df['frame_id']==10][['x', 'y']].values[0]
            agent_yaw = agent_df[agent_df['frame_id']==10]['vel_psi_rad'].values
            agent_hist_raw = agent_df[agent_df['frame_id']<=10][['x', 'y', 'vx', 'vy']].values
            agent_hist_trans = agent_df[agent_df['frame_id']<=10][['x', 'y', 'vx', 'vy']].values - np.insert(agent_pos, 2, [0,0])
            agent_fut_raw = agent_df[agent_df['frame_id']>10][['x', 'y']].values
            agent_fut_trans = agent_df[agent_df['frame_id']>10][['x', 'y']].values - agent_pos

            if agent_hist_trans.shape[0]<10:
                agent_hist_trans = np.pad(agent_hist_trans, pad_width=((10-agent_hist_trans.shape[0], 0), (0,0)), mode='edge')
                agent_hist_raw = np.pad(agent_hist_raw, pad_width=((10-agent_hist_raw.shape[0], 0), (0,0)), mode='edge')
            agent_hist_trans[:, 0], agent_hist_trans[:, 1] = self.Srotate(agent_yaw, agent_hist_trans[:, 0], agent_hist_trans[:, 1], 0, 0)
            agent_hist_trans[:, 2], agent_hist_trans[:, 3] = self.Srotate(agent_yaw, agent_hist_trans[:, 2], agent_hist_trans[:, 3], 0, 0)
            
            is_target = True
            if agent_fut_trans.shape[0]<30:
                is_target = False
                agent_fut_trans = np.pad(agent_fut_trans, pad_width=((0, 30-agent_fut_trans.shape[0]), (0,0)), mode='empty')
                agent_fut_raw = np.pad(agent_fut_raw, pad_width=((0, 30-agent_fut_raw.shape[0]), (0,0)), mode='empty')
            agent_fut_trans[:, 0], agent_fut_trans[:, 1] = self.Srotate(agent_yaw, agent_fut_trans[:, 0], agent_fut_trans[:, 1], 0, 0)

            ##

            node_feats[i] = torch.from_numpy(agent_hist_trans)
            hist_raw[i] = torch.from_numpy(agent_hist_raw)

            ## current state
            agent_state = agent_df[agent_df['frame_id']==10][['x', 'y', 'vx', 'vy', 'vel_psi_rad']].values[0]
            agent_state = torch.from_numpy(agent_state)
            agent_type = agent_df['agent_type'].values[0]

            gt[i] = torch.from_numpy(agent_fut_trans)
            future_raw[i] = torch.from_numpy(agent_fut_raw)
            target_mask.append(is_target)
            if agent_type == 'car' and is_target:
                car_target_mask.append(True)
            else:
                car_target_mask.append(False)

            if agent_type == 'car':
                car_mask.append(True)
                ped_mask.append(False)
            elif agent_type == 'pedestrian/bicycle':
                car_mask.append(False)
                ped_mask.append(True)
            else:
                print(f'\n\n check the agent type {agent_type}\n\n')

            current_df = dataframe[dataframe['frame_id']==10]
            range_x = np.square(current_df[['x']].values - agent_pos[0])
            range_y = np.square(current_df[['y']].values - agent_pos[1])
            current_df.insert(len(current_df.columns), 'dist2tar', np.sqrt(range_x + range_y), False)
            neighbor_ids = current_df[(current_df['dist2tar']>0.01) & (current_df['dist2tar']<= 30)]['track_id'].values
            ##

            node_id = node_agent_idmap[agent_id]

            # self loop
            edge_sl_feat = torch.tensor([[node_id], [node_id]])
            edge_sl_attr = torch.tensor([0, 0, 0, 0, 0]).float().unsqueeze(dim=1)
            edge_sl_type = torch.cat((self.node_type_to_indicator_vec[agent_type], self.node_type_to_indicator_vec[agent_type]), dim=1)
            edge_feats = torch.cat((edge_feats, edge_sl_feat), dim=1)
            edge_attrs = torch.cat((edge_attrs, edge_sl_attr), dim=1)
            edge_types = torch.cat((edge_types, edge_sl_type.transpose(0,1)), dim=1)

            for neighbor_id in neighbor_ids:
                neighbor_node_id = node_agent_idmap[neighbor_id]
                neighbor_state = current_df[current_df['track_id']==neighbor_id][['x', 'y', 'vx', 'vy', 'vel_psi_rad']].values[0]
                neighbor_state = torch.from_numpy(neighbor_state)
                neighbor_type = current_df[current_df['track_id']==neighbor_id]['agent_type'].values[0]
                
                edge_nb_feat = torch.tensor([[neighbor_node_id], [node_id]])
                edge_nb_attr = neighbor_state - agent_state
                edge_nb_attr = edge_nb_attr.float().unsqueeze(dim=1)
                edge_nb_type = torch.cat((self.node_type_to_indicator_vec[neighbor_type], self.node_type_to_indicator_vec[agent_type]), dim=1)
                edge_feats = torch.cat((edge_feats, edge_nb_feat), dim=1)
                edge_attrs = torch.cat((edge_attrs, edge_nb_attr), dim=1)
                edge_types = torch.cat((edge_types, edge_nb_type.transpose(0,1)), dim=1)
            ##

            # map
            map_attr = agent_state.float() - torch.cat((map_ctr.float(), torch.tensor([0., 0., 0.])), dim=0)
            map_attr = map_attr.unsqueeze(dim=1)
            map_attrs = torch.cat((map_attrs, map_attr), dim=1)


        target_mask = torch.tensor(target_mask)
        car_target_mask = torch.tensor(car_target_mask)
        car_mask = torch.tensor(car_mask)
        ped_mask = torch.tensor(ped_mask)
        
        pyg_data = Data(x=node_feats, y=gt, 
                        edge_index=edge_feats, edge_attr=edge_attrs, edge_type=edge_types, veh_map_attr=map_attrs,
                        tar_mask=target_mask, veh_tar_mask=car_target_mask, veh_mask=car_mask, ped_mask=ped_mask,
                        raw_hists=hist_raw, raw_futs=future_raw, num_agent=num_agent)

        if torch.sum(car_target_mask)>1:
            pyg_data_name = f'{self.root_path}/processed/{self.stage}/{scene}_{case}.pyg'
            torch.save(pyg_data, pyg_data_name)
    
    def process_all(self):
        for csv_id in range(len(self.csv_names)):
            scene_name = '_'.join(self.csv_names[csv_id].split('_')[:4])
            print(f'Processing Scene: {scene_name}')

            self.cur_csv_name = self.csv_names[csv_id]
            self.cur_map_path = f"{self.root_path}/maps/{'_'.join(self.cur_csv_name.split('_')[0:4])}.osm"
            self.cur_map_png_path = f"{self.root_path}/maps_png/{'_'.join(self.cur_csv_name.split('_')[0:4])}_map.png"
            map_png_image =  Image.open(self.cur_map_png_path)
            gray_image = ImageOps.grayscale(map_png_image)
            gray_image = gray_image.resize((358, 238))
            self.map_img_np = np.asarray(gray_image)
            self.map_info = self.plot_map()
            map_save_path = os.path.join(self.root_path, 'processed', self.stage, scene_name+'_map.pt')
            torch.save(torch.from_numpy(np.copy(self.map_img_np)), map_save_path)
            
            scene_df = pd.read_csv(os.path.join(self.root_path, self.stage, self.cur_csv_name))
            vx_all = scene_df['vx'].values
            vy_all = scene_df['vy'].values
            vel_psi_rad = np.around(np.arctan2(vy_all, vx_all), decimals=3)
            scene_df.insert(len(scene_df.columns), 'vel_psi_rad', vel_psi_rad, allow_duplicates=False)

            num_cases = scene_df['case_id'].values.max().astype('int')
            for case_id in range(1, num_cases+1):
                case_df = scene_df[scene_df['case_id']==case_id]
                self.generate_pyg(case_df, scene_name, case_id)

        
def parse_args(cmd_args):
    """ Parse arguments from command line input
    """
    parser = argparse.ArgumentParser(description='data preprocessing parameters')
    parser.add_argument('--root', type=str, default='./Dataset/', help="the data root dir")
    parser.add_argument('--stage', type=str, default='train', help="the dataset split")
    parser.set_defaults(render=False)
    return parser.parse_args(cmd_args)


if __name__ == '__main__':
    # Parse arguments
    args = sys.argv[1:]
    args = parse_args(args)

    csv_names = os.listdir(f'{args.root}/{args.stage}/')
    
    save_path = f'{args.root}/processed/{args.stage}' 
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    data_preprocess = IT_DATA_PRE(csv_names, save_path, args.root, args.stage)
    data_preprocess.process_all()
