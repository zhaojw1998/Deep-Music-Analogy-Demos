import json
import torch
import os
import numpy as np
from model import VAE
from data_loader import MusicArrayLoader
from torch import optim
from torch.distributions import kl_divergence, Normal
from torch.nn import functional as F
from torch.optim.lr_scheduler import ExponentialLR
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import time


class MinExponentialLR(ExponentialLR):
    def __init__(self, optimizer, gamma, minimum, last_epoch=-1):
        self.min = minimum
        super(MinExponentialLR, self).__init__(optimizer, gamma, last_epoch=-1)

    def get_lr(self):
        return [
            max(base_lr * self.gamma**self.last_epoch, self.min)
            for base_lr in self.base_lrs
        ]

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def std_normal(shape):
    N = Normal(torch.zeros(shape), torch.ones(shape))
    if torch.cuda.is_available():
        N.loc = N.loc.cuda()
        N.scale = N.scale.cuda()
    return N


def loss_function(recon,
                  recon_rhythm,
                  target_tensor,
                  rhythm_target,
                  distribution_1,
                  distribution_2,
                  beta=.1):
    CE1 = F.nll_loss(
        recon.view(-1, recon.size(-1)),
        target_tensor,
        reduction='elementwise_mean')
    CE2 = F.nll_loss(
        recon_rhythm.view(-1, recon_rhythm.size(-1)),
        rhythm_target,
        reduction='elementwise_mean')
    normal1 = std_normal(distribution_1.mean.size())
    normal2 = std_normal(distribution_2.mean.size())
    KLD1 = kl_divergence(distribution_1, normal1).mean()
    KLD2 = kl_divergence(distribution_2, normal2).mean()
    return CE1 + CE2 + beta * (KLD1 + KLD2)


def train(model, train_dataloader, epoch, loss_function, optimizer, writer, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    for step, (batch, c) in enumerate(train_dataloader):
        data_time.update(time.time() - end)
        #print(batch.shape, c.shape) # batch: batch* n_step* roll_dims; c: batch* n_step* condition_dims
        encode_tensor = batch.float()
        c = c.float()
        rhythm_target = np.expand_dims(batch[:, :, :-2].sum(-1), -1)    #batch* n_step* 1
        rhythm_target = np.concatenate((rhythm_target, batch[:, :, -2:]), -1)   #batch* n_step* 3
        rhythm_target = torch.from_numpy(rhythm_target).float()
        rhythm_target = rhythm_target.view(-1, rhythm_target.size(-1)).max(-1)[1]
        target_tensor = encode_tensor.view(-1, encode_tensor.size(-1)).max(-1)[1]
        if torch.cuda.is_available():
            encode_tensor = encode_tensor.cuda()
            target_tensor = target_tensor.cuda()
            rhythm_target = rhythm_target.cuda()
            c = c.cuda()

        optimizer.zero_grad()
        recon, recon_rhythm, dis1m, dis1s, dis2m, dis2s = model(encode_tensor, c)
        distribution_1 = Normal(dis1m, dis1s)
        distribution_2 = Normal(dis2m, dis2s)
        loss = loss_function(
            recon,
            recon_rhythm,
            target_tensor,
            rhythm_target,
            distribution_1,
            distribution_2,
            beta=args['beta'])
        loss.backward()
        losses.update(loss.item())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()

        if (step + 1) % args['display'] == 0:
            print('-------------------------------------------------------')
            for param in optimizer.param_groups:
                print('lr1: ', param['lr'])
            print_string = 'Epoch: [{0}][{1}/{2}]'.format(epoch, step + 1, len(train_dataloader))
            print(print_string)
            print_string = 'data_time: {data_time:.3f}, batch time: {batch_time:.3f}'.format(
                data_time=data_time.val,
                batch_time=batch_time.val)
            print(print_string)
            print_string = 'loss: {loss:.5f}'.format(loss=losses.avg)
            print(print_string)
        writer.add_scalar('batch_loss', losses.avg, epoch)

def main():
    # some initialization
    with open('code/model_config.json') as f:
        args = json.load(f)
    if not os.path.isdir('log'):
        os.mkdir('log')
    if not os.path.isdir('params'):
        os.mkdir('params')
    save_path = 'params/{}.pt'.format(args['name'])
    writer = SummaryWriter('log')
    model = VAE(130, args['hidden_dim'], 3, 12, args['pitch_dim'],
                args['rhythm_dim'], args['time_step'])
    if args['if_parallel']:
        model = torch.nn.DataParallel(model, device_ids=[0, 1])
        gpu_num = 2
    else:
        gpu_num = 1
    optimizer = optim.Adam(model.parameters(), lr=args['lr'])
    if args['decay'] > 0:
        scheduler = MinExponentialLR(optimizer, gamma=args['decay'], minimum=1e-5)
    if torch.cuda.is_available():
        print('Using: ', torch.cuda.get_device_name(torch.cuda.current_device()))
        model.cuda()
    else:
        print('CPU mode')
    dl = DataLoader(MusicArrayLoader(args['data_path']),
                                    batch_size=args['batch_size']*gpu_num,
                                    shuffle=True,
                                    num_workers=args['num_workers'],
                                    drop_last=True)
    # end of initialization
    model.train()
    for epoch in range(args['n_epochs']):
        train(model, dl, epoch, loss_function, optimizer, writer, args)
        if args['decay'] > 0:
            scheduler.step()
        torch.save(model.cpu().state_dict(), save_path)
        model.cuda()

if __name__ == '__main__':
    main()