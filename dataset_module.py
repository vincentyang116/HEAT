import os
import os.path as osp
import torch
from torch_geometric.data import Dataset, DataLoader

class IT_ALL_MTP_dataset(Dataset):
    def __init__(self, data_path, data_split, hist_gap=1, fut_gap=1):
        'Initialization'
        super(IT_ALL_MTP_dataset).__init__()
        self.data_path = data_path
        self.data_split = data_split
        self.hist_gap = hist_gap
        self.fut_gap = fut_gap

        self.data_names = os.listdir(f'{self.data_path}/processed/{self.data_split}')
        self.data_names = [(pyg if pyg.endswith('pyg') else None) for pyg in self.data_names]
        self.data_names = list(filter(None, self.data_names))
        print(f'Total {self.data_split} data: {self.__len__()}')

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data_names)
    
    def __getitem__(self, index):
        'Generates one sample of data'
        sample = self.data_names[index]
        data_item = torch.load(osp.join(self.data_path, 'processed', self.data_split, sample))
        data_item.edge_attr = data_item.edge_attr.transpose(0,1)
        data_item.edge_type = data_item.edge_type.transpose(0,1)
        data_item.veh_map_attr = data_item.veh_map_attr.transpose(0,1)

        data_item.x = data_item.x[:, ::self.hist_gap, :]
        data_item.y = data_item.y[:, self.fut_gap-1::self.fut_gap, :]

        map_name = '_'.join(sample.split('_')[0:4])+'_map.pt' 
        data_item.map = torch.load(f'{self.data_path}/processed/{self.data_split}/{map_name}').unsqueeze(dim=0).unsqueeze(dim=0)

        return data_item

if __name__ == '__main__':
    dataset = IT_ALL_MTP_dataset(data_path='./Dataset', data_split='test')

    print('there are {} data in this dataset'.format(dataset.__len__()))

    dataloader = DataLoader(dataset, batch_size=32,)

    for ind, test_item in enumerate(dataloader):
        print(f'x with shape {test_item.x.shape}')# : {test_item.x}') 
        print(f'y with shape {test_item.y.shape}')# : {test_item.y}')
        print(f'edge_index with shape {test_item.edge_index.shape}')#: {test_item.edge_index}
        print(f'edge_attr with shape {test_item.edge_attr.shape}')#: {test_item.edge_attr}
        print(f'edge_type with shape {test_item.edge_type.shape}')#: {test_item.edge_type}
        print(f'veh_map_attr with shape {test_item.veh_map_attr.shape}')#: {test_item.veh_map_attr}
        print(f'tar_mask with shape {test_item.tar_mask.shape}')# : {test_item.tar_mask}
        print(f'veh_tar_mask with shape {test_item.veh_tar_mask.shape}')#: {test_item.veh_tar_mask}
        print(f'veh_mask with shape {test_item.veh_mask.shape}')#: {test_item.veh_mask}
        print(f'ped_mask with shape {test_item.ped_mask.shape}')#: {test_item.ped_mask}
        print(f'raw_hists with shape {test_item.raw_hists.shape}')# : {test_item.raw_hists}')#
        print(f'raw_futs with shape {test_item.raw_futs.shape}')# : {test_item.raw_futs}')#
        print(f'map with shape {test_item.map.shape}')#: {test_item.map}')
        print('='*30)
