import torch

def Interaction_TDE_Loss(fut_pre, fut_gt, err_type='ALL', ret=None):
    if err_type == 'challenge_ADE':
        return torch.mean(torch.sum(torch.nn.MSELoss(reduction='none')(fut_pre, fut_gt), dim=2))

    TDE = torch.mean(torch.sqrt(torch.sum(torch.nn.MSELoss(reduction='none')(fut_pre, fut_gt), dim=2)), dim=0)

    if err_type == 'ALL':
        ret['ADE'] += torch.mean(TDE).item()
        ret['FDE'] += TDE[-1].item()
        ret['ApFDE'] += (torch.mean(TDE) + 0.5 * TDE[-1]).item()
        ret['LogCosh'] += torch.mean(torch.sum(torch.log(torch.cosh(fut_pre - fut_gt)), dim=2)).item()
        ret['AFDE'] += (0.5*torch.mean(TDE) + 0.5*TDE[-1]).item()
        ret['challenge_ADE'] += torch.mean(torch.sum(torch.nn.MSELoss(reduction='none')(fut_pre, fut_gt), dim=2)).item()
        return
    elif err_type == 'ADE':
        return torch.mean(TDE)
    elif err_type == 'FDE':
        return TDE[-1]
    elif err_type == 'ApFDE':
        return (torch.mean(TDE) + 0.5 * TDE[-1])
    elif err_type == 'LogCosh':
        log_cosh_loss = torch.sum(torch.log(torch.cosh(fut_pre - fut_gt)), dim=2)
        return torch.mean(log_cosh_loss)
    elif err_type == 'AFDE':
        return (0.5*torch.mean(TDE) + 0.5*TDE[-1])
    else:
        print('\n select an error type: ADE, TDE, or FDE!\n')