from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os

import json
import torch
import torch.utils.data
from torchvision.transforms import transforms as T
from opts import opts
from models.model import create_model, load_model, save_model
from datasets.dataset_factory import get_dataset
from trains.train_factory import train_factory

import torch.distributed as dist
import torch.multiprocessing as mp

from logger import Logger


def main():
    opt = opts().parse()
    print(opt)
    # logger = Logger(opt)
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
    # torch.backends.cudnn.deterministic = True
    
    # logger.write('You have chosen to seed training. '
    #                   'This will turn on the CUDNN deterministic setting, '
    #                   'which can slow down your training considerably! '
    #                   'You may see unexpected behavior when restarting '
    #                   'from checkpoints.')
    
    
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    # os.environ['NCCL_DEBUG'] ='INFO'
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

    opt.distributed = opt.world_size > 1 or opt.multiprocessing_distributed

    ngpus_per_node = len(opt.gpus) #torch.cuda.device_count()
    if opt.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        opt.world_size = ngpus_per_node * opt.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, opt))
    else:
        # Simply call main_worker function
        main_worker(opt.gpu, ngpus_per_node, opt)


def main_worker(gpu, ngpus_per_node, opt):
    opt.gpu = gpu
    logger = Logger(opt)

    if opt.gpu is not None:
        print("Use GPU: {} for training".format(opt.gpu))

    if opt.distributed:
        if opt.dist_url == "env://" and opt.rank == -1:
            opt.rank = int(os.environ["RANK"])
        if opt.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            opt.rank = opt.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=opt.dist_backend, init_method=opt.dist_url,
                                world_size=opt.world_size, rank=opt.rank)

    Dataset = get_dataset(opt.dataset, opt.task)
    f = open(opt.data_cfg)
    data_config = json.load(f)
    trainset_paths = data_config['train']
    dataset_root = data_config['root']
    f.close()
    transforms = T.Compose([T.ToTensor()])
    dataset = Dataset(opt, dataset_root, trainset_paths,
                      (1088, 608), augment=True, transforms=transforms)
                      
    opt = opts().update_dataset_info_and_set_heads(opt, dataset)

    print('Creating model...')
    model = create_model(  opt.arch, opt.heads, opt.head_conv)
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    start_epoch = 0


    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif opt.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if opt.gpu is not None:
            torch.cuda.set_device(opt.gpu)
            model.cuda(opt.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            opt.batch_size = int(opt.batch_size / ngpus_per_node)
            opt.num_workers = int((opt.num_workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif opt.gpu is not None:
        torch.cuda.set_device(opt.gpu)
        model = model.cuda(opt.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if opt.arch.startswith('alexnet') or opt.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    
    # Get dataloader

    if opt.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        train_sampler = None
   
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)


    print('Starting training...')
    Trainer = train_factory[opt.task]
    trainer = Trainer(opt, model, optimizer)
    trainer.set_device([opt.gpu], opt.chunk_sizes, opt.device)

    if opt.load_model != '':
        model, optimizer, start_epoch = load_model(
            model, opt.load_model, trainer.optimizer, opt.resume, opt.lr, opt.lr_step)

    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        if opt.distributed:
            train_sampler.set_epoch(epoch * opt.seed) # set this to a larger seed
        mark = epoch if opt.save_all else 'last'
        log_dict_train, _ = trainer.train(epoch, train_loader)
        logger.write('epoch: {} |'.format(epoch))
        for k, v in log_dict_train.items():
            logger.scalar_summary('train_{}'.format(k), v, epoch)
            logger.write('{} {:8f} | '.format(k, v))

        if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)),
                       epoch, model, optimizer)
        else:
            save_model(os.path.join(opt.save_dir, 'model_last.pth'),
                       epoch, model, optimizer)
        logger.write('\n')
        if epoch in opt.lr_step:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                       epoch, model, optimizer)
            lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
            print('Drop LR to', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        if epoch % 5 == 0 or epoch >= 25:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                       epoch, model, optimizer)
    logger.close()


if __name__ == '__main__':
    main()