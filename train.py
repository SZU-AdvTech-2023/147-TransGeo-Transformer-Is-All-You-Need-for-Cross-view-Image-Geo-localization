import argparse
import warnings
import torch
import os
import time
import shutil
import math
from datetime import datetime
import torch.multiprocessing as mp
import builtins
import torch.distributed as dist
import numpy as np

from model.TransGeo import TransGeo
from criterion.info_nce import InfoNCE
from criterion.sam import SAM
import torch.backends.cudnn as cudnn

from dataset.VIGOR import VIGOR
from dataset.CVUSA import CVUSA
from dataset.CVACT import CVACT
from dataset.global_sampler import DistributedMiningSampler,DistributedMiningSamplerVigor

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12345"


best_acc1 = 0

def main():
    args = parser.parse_args()

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
        
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()

    # print(args.distributed)
    if args.distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)

        

def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu
    args.ngpus_per_node = ngpus_per_node

    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.distributed:
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method="env://",
                                world_size=args.world_size, rank=args.rank)
        
    print("=> creating model '{}'")

    model = TransGeo(args=args)

    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)

            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))

    # criterion = SoftTripletBiLoss().cuda(args.gpu)
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion = InfoNCE(loss_function=loss_fn, device=args.gpu).cuda(args.gpu)

    if args.op == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    elif args.op == 'adam':
        optimizer = torch.optim.Adam(parameters, args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)
    elif args.op == 'adamw':
        optimizer = torch.optim.AdamW(parameters, args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay, amsgrad=False)
    elif args.op == 'sam':
        base_optimizer = torch.optim.AdamW
        optimizer = SAM(parameters, base_optimizer,  lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay, amsgrad=False, rho=args.rho, adaptive=args.asam)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            if not args.crop:
                args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.crop and args.sat_res != 0:
                # pos_embed_reshape的形状应该是：1*7*38*384
                pos_embed_reshape = checkpoint['state_dict']['module.reference_net.pos_embed'][:, 2:, :].reshape(
                    [1, 7, 38, model.module.reference_net.embed_dim]).permute((0, 3, 1, 2))
                
                checkpoint['state_dict']['module.reference_net.pos_embed'] = \
                    torch.cat([checkpoint['state_dict']['module.reference_net.pos_embed'][:, :2, :],
                               torch.nn.functional.interpolate(pos_embed_reshape, (
                               140 // model.module.reference_net.patch_embed.patch_size[0],
                               770 // model.module.reference_net.patch_embed.patch_size[1]),
                                                               mode='bilinear').permute((0, 2, 3, 1)).reshape(
                                   [1, -1, model.module.reference_net.embed_dim])], dim=1)

            model.load_state_dict(checkpoint['state_dict'])
            if args.op == 'sam' and args.dataset != 'cvact':    # Loading the optimizer status gives better result on CVUSA, but not on CVACT.
                optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # 加载数据集
    if not args.multiprocessing_distributed or args.gpu == 0:
        if not os.path.exists(args.save_path):
            os.mkdir(args.save_path)
            os.mkdir(os.path.join(args.save_path, 'attention'))
            os.mkdir(os.path.join(args.save_path, 'attention','train'))
            os.mkdir(os.path.join(args.save_path, 'attention','val'))
    
    if args.dataset == 'vigor':
        dataset = VIGOR
        mining_sampler = DistributedMiningSamplerVigor
    elif args.dataset == 'cvusa':
        dataset = CVUSA
        mining_sampler = DistributedMiningSampler
    elif args.dataset == 'cvact':
        dataset = CVACT
        mining_sampler = DistributedMiningSampler

    train_dataset = dataset(mode='train', print_bool=True, same_area=(not args.cross),args=args)
    train_scan_dataset = dataset(mode='scan_train' if args.dataset == 'vigor' else 'train', print_bool=True, same_area=(not args.cross), args=args)
    val_scan_dataset = dataset(mode='scan_val', same_area=(not args.cross), args=args)
    val_query_dataset = dataset(mode='test_query', same_area=(not args.cross), args=args)
    val_reference_dataset = dataset(mode='test_reference', same_area=(not args.cross), args=args)

    if args.distributed:
        if args.mining:
            train_sampler = mining_sampler(train_dataset, batch_size=args.batch_size, dim=args.dim, save_path=args.save_path)
            if args.resume:
                train_sampler.load(args.resume.replace(args.resume.split('/')[-1],''))
                # print(args.resume.replace(args.resume.split('/')[-1],''))
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    train_scan_loader = torch.utils.data.DataLoader(
        train_scan_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=torch.utils.data.distributed.DistributedSampler(train_scan_dataset), drop_last=False)

    val_scan_loader = torch.utils.data.DataLoader(
        val_scan_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        sampler=torch.utils.data.distributed.DistributedSampler(val_scan_dataset), drop_last=False)

    val_query_loader = torch.utils.data.DataLoader(
        val_query_dataset,batch_size=32, shuffle=False,
        num_workers=args.workers, pin_memory=True) # 512, 64
    
    val_reference_loader = torch.utils.data.DataLoader(
        val_reference_dataset, batch_size=64, shuffle=False,
        num_workers=args.workers, pin_memory=True) # 80, 128
    
    if args.evaluate:
        if not args.multiprocessing_distributed or args.gpu == 0:
            validate(val_query_loader, val_reference_loader, model, args)
        return

    for epoch in range(args.start_epoch, args.epochs):

        print('start epoch:{}, date:{}'.format(epoch, datetime.now()))
        if args.distributed:
            train_sampler.set_epoch(epoch)
            if args.mining:
                train_sampler.update_epoch()
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, train_sampler)

        # evaluate on validation set
        if not args.multiprocessing_distributed or args.gpu == 0:
            acc1 = validate(val_query_loader, val_reference_loader, model, args)
            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
        torch.distributed.barrier()

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, is_best, filename='checkpoint.pth.tar', args=args) # 'checkpoint_{:04d}.pth.tar'.format(epoch)

    if not args.crop:
        model.module.reference_net.save = os.path.join(args.save_path, 'attention', 'train')
        scan(train_scan_loader, model, args)
        model.module.reference_net.save = os.path.join(args.save_path, 'attention', 'val')
        scan(val_scan_loader, model, args)


def train(train_loader, model, criterion, optimizer, epoch, args, train_sampler=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    mean_ps = AverageMeter('Mean-P', ':6.2f')
    mean_ns = AverageMeter('Mean-N', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, mean_ps, mean_ns],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images_q, images_k, indexes, _, delta, atten) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images_q = images_q.cuda(args.gpu, non_blocking=True)
            images_k = images_k.cuda(args.gpu, non_blocking=True)
            indexes = indexes.cuda(args.gpu, non_blocking=True)

        # compute output
        if args.crop:
            embed_q, embed_k = model(im_q=images_q, im_k=images_k, delta=delta, atten=atten)
        else:
            embed_q, embed_k = model(im_q =images_q, im_k=images_k, delta=delta)

        # loss, mean_p, mean_n = criterion(embed_q, embed_k)
        loss = criterion(embed_q, embed_k, model.module.logit_scale)


        if args.mining:
            train_sampler.update(concat_all_gather(embed_k).detach().cpu().numpy(),concat_all_gather(embed_q).detach().cpu().numpy(),concat_all_gather(indexes).detach().cpu().numpy())
        losses.update(loss.item(), images_q.size(0))
        # mean_ps.update(mean_p, images_q.size(0))
        # mean_ns.update(mean_n, images_q.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        if args.op != 'sam':
            optimizer.step()
        else:
            optimizer.first_step(zero_grad=True)

            # second forward-backward pass, only for ASAM
            if args.crop:
                embed_q, embed_k = model(im_q=images_q, im_k=images_k, delta=delta, atten=atten)
            else:
                embed_q, embed_k = model(im_q=images_q, im_k=images_k, delta=delta)

            # loss, mean_p, mean_n = criterion(embed_q, embed_k)
            loss = criterion(embed_q, embed_k, model.module.logit_scale)
            loss.backward()
            optimizer.second_step(zero_grad=True)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

        del loss
        del embed_q
        del embed_k

# save all the attention map
def scan(loader, model, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    progress = ProgressMeter(
        len(loader),
        [batch_time, data_time],
        prefix="Scan:")

    # switch to eval mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images_q, images_k, _, indexes , delta, _) in enumerate(loader):

            # measure data loading time
            data_time.update(time.time() - end)

            if args.gpu is not None:
                images_q = images_q.cuda(args.gpu, non_blocking=True)
                images_k = images_k.cuda(args.gpu, non_blocking=True)
                indexes = indexes.cuda(args.gpu, non_blocking=True)

            # compute output
            embed_q, embed_k = model(im_q =images_q, im_k=images_k, delta=delta, indexes=indexes)

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)


# query features and reference features should be computed separately without correspondence label
def validate(val_query_loader, val_reference_loader, model, args):
    batch_time = AverageMeter('Time', ':6.3f')
    progress_q = ProgressMeter(
        len(val_query_loader),
        [batch_time],
        prefix='Test_query: ')
    progress_k = ProgressMeter(
        len(val_reference_loader),
        [batch_time],
        prefix='Test_reference: ')

    # switch to evaluate mode
    model_query = model.module.query_net
    model_reference = model.module.reference_net
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model_query.cuda(args.gpu)
            model_reference.cuda(args.gpu)

    model_query.eval()
    model_reference.eval()
    print('model validate on cuda', args.gpu)

    query_features = np.zeros([len(val_query_loader.dataset), args.dim])
    query_labels = np.zeros([len(val_query_loader.dataset)])
    reference_features = np.zeros([len(val_reference_loader.dataset), args.dim])

    with torch.no_grad():
        end = time.time()
        # reference features
        for i, (images, indexes, atten) in enumerate(val_reference_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                indexes = indexes.cuda(args.gpu, non_blocking=True)

            # compute output
            if args.crop:
                reference_embed = model_reference(x=images, atten=atten)
            else:
                reference_embed = model_reference(x=images, indexes=indexes)  # delta

            reference_features[indexes.cpu().numpy().astype(int), :] = reference_embed.detach().cpu().numpy()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress_k.display(i)

        end = time.time()

        # query features
        for i, (images, indexes, labels) in enumerate(val_query_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                indexes = indexes.cuda(args.gpu, non_blocking=True)
                labels = labels.cuda(args.gpu, non_blocking=True)

            # compute output
            query_embed = model_query(images)

            query_features[indexes.cpu().numpy(), :] = query_embed.cpu().numpy()
            query_labels[indexes.cpu().numpy()] = labels.cpu().numpy()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress_q.display(i)

        [top1, top5] = accuracy(query_features, reference_features, query_labels.astype(int))

    if args.evaluate:
        np.save(os.path.join(args.save_path, 'grd_global_descriptor.npy'), query_features)
        np.save('sat_global_descriptor.npy', reference_features)

    return top1


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', args=None):
    torch.save(state, os.path.join(args.save_path,filename))
    if is_best:
        shutil.copyfile(os.path.join(args.save_path,filename), os.path.join(args.save_path,'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(query_features, reference_features, query_labels, topk=[1,5,10]):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    ts = time.time()
    N = query_features.shape[0]
    M = reference_features.shape[0]
    topk.append(M//100)
    results = np.zeros([len(topk)])
    # for CVUSA, CVACT
    if N < 80000:
        query_features_norm = np.sqrt(np.sum(query_features**2, axis=1, keepdims=True))
        reference_features_norm = np.sqrt(np.sum(reference_features ** 2, axis=1, keepdims=True))
        similarity = np.matmul(query_features/query_features_norm, (reference_features/reference_features_norm).transpose())

        for i in range(N):
            ranking = np.sum((similarity[i,:]>similarity[i,query_labels[i]])*1.)

            for j, k in enumerate(topk):
                if ranking < k:
                    results[j] += 1.
    else:
        # split the queries if the matrix is too large, e.g. VIGOR
        assert N % 4 == 0
        N_4 = N // 4
        for split in range(4):
            query_features_i = query_features[(split*N_4):((split+1)*N_4), :]
            query_labels_i = query_labels[(split*N_4):((split+1)*N_4)]
            query_features_norm = np.sqrt(np.sum(query_features_i ** 2, axis=1, keepdims=True))
            reference_features_norm = np.sqrt(np.sum(reference_features ** 2, axis=1, keepdims=True))
            similarity = np.matmul(query_features_i / query_features_norm,
                                   (reference_features / reference_features_norm).transpose())
            for i in range(query_features_i.shape[0]):
                ranking = np.sum((similarity[i, :] > similarity[i, query_labels_i[i]])*1.)
                for j, k in enumerate(topk):
                    if ranking < k:
                        results[j] += 1.

    results = results/ query_features.shape[0] * 100.
    print('Percentage-top1:{}, top5:{}, top10:{}, top1%:{}, time:{}'.format(results[0], results[1], results[2], results[-1], time.time() - ts))
    return results[:2]


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pytorch ImageNet Training')

    parser.add_argument('--workers', default=32, type=int, metavar='N',
                       help='number of data loading workers (default: 32)')
    parser.add_argument('--world-size', default=1, type=int,
                       help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--multiprocessing-distributed', action='store_true')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
    
    parser.add_argument('--dim', default=1000, type=int,
                    help='feature dimension (default: 128)')
    parser.add_argument('--dataset', default='cvusa', type=str,
                    help='vigor, cvusa, cvact')
    parser.add_argument('--op', default='sam', type=str,
                    help='sgd, adam, adamw')
    parser.add_argument('--asam', action='store_true',
                    help='asam')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
    parser.add_argument('--save_path', default='', type=str, metavar='PATH',
                    help='path to save checkpoint (default: none)')
    
    parser.add_argument('--sat_res', default=0, type=int,
                    help='resolution for satellite')
    parser.add_argument('--fov', default=0, type=int,
                    help='Fov')
    parser.add_argument('--crop', action='store_true',
                    help='nonuniform crop')
    parser.add_argument('--rho', default=0.05, type=float,
                    help='rho for sam')
    parser.add_argument('--cross', action='store_true',
                    help='use cross area')
    
    parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')
    parser.add_argument('--mining', action='store_true',
                    help='mining')
    
    parser.add_argument('--batch-size', default=256, type=int,
                        metavar='N',)
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--wd', '--weight-decay', default=0, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
    
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

    main()