from __future__ import print_function
import datetime
import sys
import argparse
import pprint
import torch
from torch.optim.lr_scheduler import MultiStepLR
from loss import Interaction_TDE_Loss
from torch_geometric.data import DataLoader
from heat_g_model import IT_Heat_Net
from heat_gir_model import IT_HeatIR_Net
from dataset_module import IT_ALL_MTP_dataset
import matplotlib.pyplot as plt

def validation(model_to_val, metrics, visualize=False):
    model_to_val.eval()
    with torch.no_grad():
        metrics_dict = {'ADE': 0.0, 'FDE': 0.0, 'ApFDE': 0.0, 'LogCosh': 0.0, 'AFDE': 0.0, 'challenge_ADE': 0.0}

        for i, data in enumerate(val_loader):
            data.x = data.x[:,-args.hist:,0:args.input_size]

            fut_pred = model_to_val(data.to(device))

            masked_gt = data.y[data.tar_mask]
            masked_pred = fut_pred[data.tar_mask]

            # if visualize:
            #     fig = plt.figure()
            #     for i in range(masked_pred.shape[0]):
            #         plt.plot(masked_pred[i,:,0].cpu(), masked_pred[i,:,1].cpu(),'bo')
            #         plt.plot(masked_gt[i,:,0].cpu(), masked_gt[i,:,1].cpu(),'ro')
            #     plt.show()

            if metrics == 'ALL':
                Interaction_TDE_Loss(masked_pred, masked_gt, err_type=metrics, ret=metrics_dict)
            else:
                metrics_dict[metrics] += Interaction_TDE_Loss(masked_pred, masked_gt, err_type=metrics).item() 

        for k,v in metrics_dict.items():
            metrics_dict[k] = round(v / i, 4)         

    print('Evaluation results: ', metrics_dict)
    return 

def parse_args(args):
    """ Parse arguments from command line input
    """
    parser = argparse.ArgumentParser(description='Training parameters')
    parser.add_argument('--network', type=str, default='HeatIR', help="the model structure")
    parser.add_argument('--gnn', type=str, default='GAT', help="the GNN to be used")
    parser.add_argument('--rnn', type=str, default='GRU', help="the RNN to be used")
    parser.add_argument('--input_size', type=int, default=4, help="the Number of data to be used")
    parser.add_argument('--hist', type=int, default=10, help="length of history 10, 30, 50")
    parser.add_argument('--future', type=int, default=30, help="length of future 50")
    parser.add_argument('--gpu', type=str, default='1', help="the GPU to be used")
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--epoch', type=int, default=60, help='training epoch')
    parser.add_argument('--split', type=str, default='train', help='dataset split')
    parser.add_argument('--eval', action="store_true")
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--metrics', type=str, default='ADE')

    parser.set_defaults(render=False)
    return parser.parse_args(args)

if __name__ == '__main__':
   
    pp = pprint.PrettyPrinter(indent=1)
    # Parse and set arguments
    args = sys.argv[1:]
    args = parse_args(args)

    args.encoder_size = 64 
    args.decoder_size = 256 
    args.dyn_embedding_size = 64 

    args.heat_in_channels_node = 64
    args.heat_in_channels_edge_attr = 5
    args.heat_in_channels_edge_type = 6
    args.heat_edge_attr_emb_size = 64
    args.heat_edge_type_emb_size = 64
    args.heat_node_emb_size = 64
    args.heat_out_channels = 128
    args.heat_heads = 3
    args.heat_concat=True
    args.out_length = 30
    args.input_embedding_size = 32 
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else "cpu")

    # Initialize network
    if args.network == 'Heat':
        print('Creating {} model'.format(args.network))
        train_net = IT_Heat_Net(args)
    elif args.network == 'HeatIR':
        print('Creating {} model'.format(args.network))
        train_net = IT_HeatIR_Net(args)
    else:
        print('\nselect a proper model type!\n')

    
    val_set = IT_ALL_MTP_dataset('./Dataset/', 'val')
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    
    if args.eval:
        train_net.load_state_dict(torch.load(args.model))
        train_net.to(device)
        validation(train_net, args.metrics)
        sys.exit(0)

    pp.pprint(vars(args))
    
    ## Initialize optimizer 
    train_set = IT_ALL_MTP_dataset('./Dataset/', args.split)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    train_net.to(device)
    optimizer = torch.optim.Adam(train_net.parameters(),lr=0.001) 
    scheduler = MultiStepLR(optimizer, milestones=[1, 10, 20, 30], gamma=0.5)

    bs_interval = 200
    val_interval = 5
    datestr = datetime.datetime.now().strftime('%Y%m%d%H%M')

    for ep in range(1, args.epoch+1):
        print(f'Training Epoch: {ep}...')
        train_net.train()
        train_running_loss = 0.0
        for i, data in enumerate(train_loader):
            data.x = data.x[:,:,0:args.input_size]
            if torch.sum(torch.isnan(data.x))>0:
                print('nan in x')

            optimizer.zero_grad()
            fut_pred = train_net(data.to(device))
            masked_gt = data.y[data.tar_mask]
            masked_pred = fut_pred[data.tar_mask]
            
            train_l = Interaction_TDE_Loss(masked_pred, masked_gt, err_type=args.metrics)
            if torch.sum(torch.isnan(train_l))>0:
                print('nan in train loss')

            train_l.backward()
            a = torch.nn.utils.clip_grad_norm_(train_net.parameters(), 10)
            optimizer.step()
            train_running_loss += train_l.item()
            if (i % bs_interval == 0) and (i>0):
                print(f'\tBatch {i} training {args.metrics} loss: {round(train_running_loss / i, 4)}')

        train_loss_ep = round(train_running_loss / i, 4)
        print(f'\tEpoch {ep} training {args.metrics} loss: {train_loss_ep}')
        scheduler.step()

        save_model_to_PATH = f'./models/{datestr}_{args.network}_ep{ep}.tar'
        torch.save(train_net.state_dict(), save_model_to_PATH)

        if ep % val_interval == 0:
            validation(train_net, 'ALL')