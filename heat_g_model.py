import torch
from base_model import IT_Base_Net
from heat_layer import HEATlayer

class IT_Heat_Net(IT_Base_Net):
    def __init__(self, args):
        super(IT_Heat_Net, self).__init__(args)
        self.args = args
        # HEAT layers
        self.heat_conv1 = HEATlayer(in_channels_node = self.args.heat_in_channels_node, 
                                    in_channels_edge_attr = self.args.heat_in_channels_edge_attr,
                                    in_channels_edge_type = self.args.heat_in_channels_edge_type,
                                    edge_attr_emb_size=self.args.heat_edge_attr_emb_size, 
                                    edge_type_emb_size=self.args.heat_edge_type_emb_size, 
                                    node_emb_size = self.args.heat_node_emb_size,
                                    out_channels =  self.args.heat_out_channels,
                                    heads = self.args.heat_heads,
                                    concat = self.args.heat_concat,
                                    )

        self.heat_conv2 = HEATlayer(in_channels_node = self.args.heat_out_channels+int(self.args.heat_concat)*(self.args.heat_heads-1)*self.args.heat_out_channels, 
                                    in_channels_edge_attr = self.args.heat_in_channels_edge_attr,
                                    in_channels_edge_type = self.args.heat_in_channels_edge_type,
                                    edge_attr_emb_size=self.args.heat_edge_attr_emb_size, 
                                    edge_type_emb_size=self.args.heat_edge_type_emb_size, 
                                    node_emb_size = self.args.heat_node_emb_size,
                                    out_channels =  self.args.heat_out_channels,
                                    heads = self.args.heat_heads,
                                    concat = self.args.heat_concat,
                                    )
        # fully connected
        self.nbrs_fc = torch.nn.Linear(int(self.args.heat_concat)*(self.args.heat_heads-1)*self.args.heat_out_channels + self.args.heat_out_channels, 1*self.args.heat_out_channels)
        # Decoder LSTM
        self.veh_dec_rnn = torch.nn.LSTM(self.args.encoder_size, self.args.decoder_size, 2, batch_first=True)
        self.ped_dec_rnn = torch.nn.LSTM(self.args.encoder_size, self.args.decoder_size, 2, batch_first=True)
    
    def HEAT_Interaction(self, hist_enc, edge_idx, edge_attr, edge_type, veh_node_mask, ped_node_mask):
        
        gat_feature = self.heat_conv1(hist_enc, edge_idx, edge_attr, edge_type, veh_node_mask, ped_node_mask)
        gat_feature = self.heat_conv2(gat_feature, edge_idx, edge_attr, edge_type, veh_node_mask, ped_node_mask)
        GAT_Enc = self.leaky_relu(self.nbrs_fc(gat_feature))
        return GAT_Enc
    
    def forward(self, data_pyg):
        fwd_Hist_Enc = self.RNN_Encoder(data_pyg.x, data_pyg.veh_mask, data_pyg.ped_mask)
        fwd_tar_GAT_Enc = self.HEAT_Interaction( hist_enc=fwd_Hist_Enc,
                                                edge_idx=data_pyg.edge_index, 
                                                edge_attr=data_pyg.edge_attr, 
                                                edge_type=data_pyg.edge_type.float(),
                                                veh_node_mask=data_pyg.veh_mask, 
                                                ped_node_mask=data_pyg.ped_mask)
        fut_pred = self.decode(fwd_tar_GAT_Enc, data_pyg.veh_mask, data_pyg.ped_mask)
        return fut_pred
    