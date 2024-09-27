import glob
import os
import random
import datetime

import torch
import torch.multiprocessing
from torch.utils.data import DataLoader
import socket
import wandb

from record import eval_model, output_model
from network.net_backbone import ResUNet
from utils import *
from loader import load_model, realworld_FF
from loss import RecLoss


def load_id_wandb_current(config, record_flag, pretrained_root, output_root, id=None):
    if (config is None) == (id is None):
        raise Exception('One of the two must be set.')

    if config is None:
        print(f"path {osp.join(pretrained_root, id)}")
        config = glob.glob(osp.join(pretrained_root, id, '*.yml'))[0]
        run_id = id
        print('config restored from: ', run_id)
    else:
        if len(config.split('.')) == 1:
            config = config + '.yml'
        config = os.getcwd() + '/config/' + config

    with open(config, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)
        mode = cfg.mode
        seed = cfg.randomseed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    wandb_obj = None
    if id is None:
        current_time = datetime.now().strftime('%m%d%H%M')
        run_id = f'{current_time}_{cfg.mode}'
        if record_flag:
            wandb_obj = wandb.init(project=f'MAIR-{mode}', id=run_id)
            wandb_obj.config.update(cfg)
    else:
        if record_flag:
            wandb_obj = wandb.init(project=f'MAIR-{mode}', id=run_id, resume=True)

    # path to the experimental model
    experiment = osp.join(pretrained_root, run_id)

    # save scripts and config file
    out_doc_root = osp.join(output_root, run_id)
    if record_flag and id is None:
        os.makedirs(out_doc_root, exist_ok=True)
        os.system(f'cp *.py {out_doc_root}')
        os.system(f'cp network/*.py {out_doc_root}')
        os.system(f'cp {config} {out_doc_root}')
    return cfg, run_id, wandb_obj, experiment


def load_dataloader_current(dataRoot, outputRoot, cfg, is_DDP, phase_list, debug=False):
    worker_per_gpu = cfg.num_workers
    batch_per_gpu = cfg.batchsize
    print('batch_per_gpu', batch_per_gpu, 'worker_per_gpu', worker_per_gpu)

    dict_loader = {}
    for phase in phase_list:
        sampler = None
        is_shuffle = True
        if phase == 'custom':
            if cfg.mode == 'MG':
                assert False, 'Not implemented'
                dataset = realworld_FF_singleview(dataRoot, cfg)
            else:
                dataset = realworld_FF(dataRoot, cfg, outputRoot=outputRoot)
            is_shuffle = False
        elif phase == 'mat_edit':
            assert False, 'Not implemented'
            dataset = mat_edit_dataset(dataRoot, cfg)
            is_shuffle = False
        else:
            assert False, 'Not implemented'
            dataset = OpenroomsFF(dataRoot, cfg, phase, debug)
            if phase == 'test':
                is_shuffle = False
        if is_DDP:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            is_shuffle = False

        pinned = cfg.pinned and phase == 'TRAIN'
        loader = DataLoader(dataset, batch_size=batch_per_gpu, shuffle=is_shuffle, num_workers=worker_per_gpu,
                            pin_memory=pinned, sampler=sampler)

        dict_loader[phase] = [loader, sampler]
        print(f'create dataset - mode {cfg.mode}, shuffle: {is_shuffle}')
        print(f'{phase} dataset number of sample: {dataset.length}')
        print(f'{phase} loader number of sample: {len(loader)}')
    return dict_loader


def test(gpu, num_gpu, run_mode, phase_list,
         data_root, pretrained_root, output_root,
         is_DDP=False, run_id=None, config=None, port=2958, num_K=None):
    if is_DDP:
        torch.distributed.init_process_group(backend='nccl', init_method=f'tcp://127.0.0.1:{port}', world_size=num_gpu,
                                             rank=gpu)
        torch.cuda.set_device(gpu)

    cfg, run_id, wandb_obj, experiment = load_id_wandb_current(config, False, pretrained_root, output_root, run_id)
    if run_mode == 'test':
        if cfg.mode == 'VSG':
            cfg.batchsize = 1
            cfg.num_workers = 1
        else:
            cfg.batchsize = 4
            cfg.num_workers = 3
    elif run_mode == 'output':
        cfg.batchsize = 1
        cfg.num_workers = 1

    if num_K is not None:
        cfg.num_K = num_K
    cfg.full_load = True
    dict_loaders = load_dataloader_current(data_root, output_root, cfg, is_DDP, phase_list)

    model = load_model(cfg, gpu, experiment, is_train=False, is_DDP=is_DDP, wandb_obj=wandb_obj)
    model.switch_to_eval()
    if dict_loaders is not None:
        for phase in dict_loaders:
            data_loader, _ = dict_loaders[phase]
            if run_mode == 'output':
                output_model(model, data_loader, gpu, cfg)

            if run_mode == 'test':
                cfg.losskey.append('rgb')
                cfg.losstype.append('l2')
                cfg.weight.append(1.0)
                cfg.lossmask.append('mask')
                loss_agent = RecLoss(cfg)
                eval_dict, _ = eval_model(model, data_loader, gpu, cfg, num_gpu, loss_agent, 'test', 0)
                print(eval_dict)
    if is_DDP:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    dataroot = './Examples/input_processed'
    pretrained = 'pretrained/MAIR'
    output_root = './out/'
    run_id = '05190941_VSG'
    run_mode = 'output'
    phase_list = ['custom', ]
    test(0, 1, run_mode, phase_list, dataroot, pretrained, output_root, False, run_id=run_id, config=None)
