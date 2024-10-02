import glob
import os
import random
import datetime
import argparse
import sys

import torch
import torch.multiprocessing
from torch.utils.data import DataLoader
from torch.cuda.amp.autocast_mode import autocast
import socket
import wandb
from tqdm import tqdm

import MAIR
from record import eval_model
from network.net_backbone import ResUNet
from utils import *
from loader import load_model, realworld_FF
from loss import RecLoss
from model import *
import single_view_dataset

openrooms_utils_path = "./openrooms_utils"
sys.path.append(openrooms_utils_path)
from openrooms_utils import env_util


def MAIR_new_forward(self, data, cfg, forward_mode):
    """

    Args:
        self:
        data:
            data['i']: input image, linear RGB
            data['cds_conf']: confidence map
            data['cds_dn']: depth map, normalized to [0, 1]
            data['cds_dg']: depth gradient
        cfg:
        forward_mode:

    Returns:

    """

    assert cfg.d_type == "cds" and cfg.mode == "VSG" and cfg.VSGEncoder.src_type == "exi" and forward_mode == "output", \
        "Only support cds, VSG, exi, output"

    with autocast(enabled=cfg.autocast):
        empty = torch.tensor([], device=data['i'].device)
        pred = {}
        gt = {k: data[k] for k in set(cfg.losskey) & set(data)}
        mask = data.pop('m')
        if 'e' in gt or 'e_d' in gt:
            assert False, "code not checked"
            Bn, env_rows, env_cols, _, _ = gt.get('e', gt.get('e_d')).shape
            mask = F.adaptive_avg_pool2d(mask, (env_rows, env_cols))
            if 'e' in gt:
                mask = mask * (torch.mean(gt['e'], dim=(3, 4)) > 0.001).float()[:, None]
            else:
                mask = (mask > 0.9).float()

        assert data['i'].shape[1:] == (3, 240, 320)
        n, d, _, _ = self.MGNet(data['i'], data['cds_dn'], data['cds_conf'], data['cds_dg'])

        if cfg.d_type == 'net':
            assert False, 'd_type net not implemented'
            d = d / torch.amax(d, dim=[1, 2, 3], keepdim=True)
            c = empty
        elif cfg.d_type == 'cds':
            d = data['cds_dn']
            c = data['cds_conf']
        else:
            assert False, f"cfg d_type: {cfg.d_type} not checked"
        ### end mode incident
        if cfg.mode == 'incident':
            assert False, 'mode incident not checked'
            axis, sharp, intensity, pred['vis'], pred['e_d'] = self.InLightSG(data['i'], d, c, n, empty, empty,
                                                                              empty, mode=0)
            pred['vis'] = pred['vis'][:, :, 0, :, :, None, None]
            return pred, gt, get_mask_dict('default', mask)

        if cfg.mode == 'exitant' or (cfg.mode == 'VSG' and (cfg.VSGEncoder.src_type == 'exi' or
                                                            cfg.VSGEncoder.src_type == 'train')):
            VSG_DL = self.ExDLVSG(data['i'], d, c, n)
            if cfg.mode == 'exitant':
                assert False, 'mode exitant not checked'
                cam_mat = data['cam'] / (cfg.imWidth / env_cols)
                cam_mat[:, -1, -1] = 1.0
                pixels_DL = self.pixels[:, :env_rows, :env_cols]
                depth_low = F.adaptive_avg_pool2d(d, (env_rows, env_cols)).permute([0, 2, 3, 1])
                if cfg.d_type == 'cds':
                    conf_low = F.adaptive_avg_pool2d(c, (env_rows, env_cols))
                    mask = mask * conf_low

                cam_coord = (depth_low * (torch.inverse(cam_mat[:, None, None]) @ pixels_DL)[..., 0])[..., None, :]
                normal_low = F.adaptive_avg_pool2d(n, (env_rows, env_cols)).permute([0, 2, 3, 1])
                normal_low = F.normalize(normal_low, p=2.0, dim=-1)
                N2C = get_N2C(normal_low, self.up)
                ls_rdf = (N2C.unsqueeze(-3) @ self.ls).squeeze(-1) * self.rub_to_rdf
                vsg_alpha = 0.5 * (torch.clamp(1.01 * torch.tanh(VSG_DL[:, -1]), -1, 1) + 1)
                pred['vis'] = vsg_alpha
                pred['e_d'] = envmapfromVSG(VSG_DL, cam_coord, ls_rdf, self.r_dist, data['bb'], cfg.sg_order)
                return pred, gt, get_mask_dict('env', mask[:, 0, ..., None, None])

        axis, sharp, intensity, vis = self.InLightSG(data['i'], d, c, n, empty, empty, empty, mode=1)
        featmaps = self.ContextNet(data['i'], d, c, n, empty, empty) if cfg.ContextNet.use else empty
        all_rgb, viewdir, proj_err, _ = compute_projection(self.pixels, data['all_cam'], data['c2w'],
                                                           data['all_depth'], data['all_i'])

        bn, h_, w_, v_, _ = all_rgb.shape
        n_low = F.adaptive_avg_pool2d(n, (axis.shape[-2], axis.shape[-1])).permute([0, 2, 3, 1])
        N2C = get_N2C(F.normalize(n_low, p=2.0, dim=-1), self.up)  # transformation matrix from surface coordinate to camera coordinate system
        axis_cam = torch.einsum('bhwqp,bsphw->bsqhw', N2C, axis)  # axis in camera coordinate system
        DL_flat = F.interpolate(
            torch.cat([axis_cam, sharp, intensity], dim=2).reshape(bn, -1, axis.shape[-2], axis.shape[-1]),
            scale_factor=4, mode='nearest')
        DL = rearrange(DL_flat, 'b (q p) h w  -> b h w () q p', q=cfg.InLightSG.SGNum)

        n_low = F.adaptive_avg_pool2d(n, (h_, w_)).permute([0, 2, 3, 1])
        n_low = F.normalize(n_low, p=2.0, dim=-1).unsqueeze(-2)
        brdf_feat = self.AggregationNet(all_rgb, None, None, None, proj_err, featmaps, viewdir, n_low, DL)
        if cfg.RefineNet.use:
            a, r, _ = self.RefineNet(data['i'], d, c, n, brdf_feat)
        else:
            assert False, 'code not checked'
            brdf_feat = F.interpolate(brdf_feat, scale_factor=2.0, mode='bilinear')
            a = 0.5 * (torch.clamp(1.01 * torch.tanh(brdf_feat[:, :3]), -1, 1) + 1)
            r = 0.5 * (torch.clamp(1.01 * torch.tanh(brdf_feat[:, 3:]), -1, 1) + 1)
        ### end mode BRDF
        if cfg.mode == 'BRDF':
            assert False, 'mode BRDF not checked'
            pred['a'], pred['r'] = a, r
            return pred, gt, get_mask_dict('default', mask)

        assert a.shape[0] == 1
        if forward_mode == 'output':
            env_rows, env_cols, env_w, env_h = self.envRows, self.envCols, self.env_width, self.env_height
        else:
            assert False, "Code not checked"

        source = torch.cat([data['i'], d, c, n, a, r], dim=1)
        if cfg.VSGEncoder.src_type == 'exi' or cfg.VSGEncoder.src_type == 'train':
            VSG_DL = torch.clamp(1.01 * torch.tanh(VSG_DL), -1, 1)
            vsg_tmp1 = 0.5 * (F.normalize(VSG_DL[:, :3], p=2.0, dim=1) + 1)
            vsg_tmp2 = 0.5 * (VSG_DL[:, 3:] + 1)
            VSG_DL = torch.cat([vsg_tmp1, vsg_tmp2], dim=1)
            del vsg_tmp1, vsg_tmp2

        elif cfg.VSGEncoder.src_type == 'inci':
            assert False, 'VSGEncoder src_type inci not implemented'
            DL_flatten = F.interpolate(DL_flat, scale_factor=2, mode='nearest')
            source = torch.cat([source, DL_flatten], dim=1)
            VSG_DL = None
        elif cfg.VSGEncoder.src_type == 'none':
            assert False, 'VSGEncoder src_type none not implemented'
            VSG_DL = torch.zeros([1, 8, 32, 32, 32], dtype=a.dtype, device=a.device)
        else:
            assert False, "Code not checked"
        vsg_in = get_visible_surface_volume(data['voxel_grid_front'], source, data['cam'])
        vsg = self.VSGEncoder(vsg_in, VSG_DL)  # normal
        vsg_alpha = 0.5 * (torch.clamp(1.01 * torch.tanh(vsg[:, -1]), -1, 1) + 1)
        pred['vis'] = vsg_alpha

        a_low = F.adaptive_avg_pool2d(a, (env_rows, env_cols))
        r_low = F.adaptive_avg_pool2d(r, (env_rows, env_cols))
        d_low = F.adaptive_avg_pool2d(d, (env_rows, env_cols))
        N2C = get_N2C(n_low.squeeze(-2), self.up)
        cam_mat = data['cam'] / (cfg.imWidth / env_cols)
        cam_mat[:, -1, -1] = 1.0
        pixels = self.pixels[:, :env_rows, :env_cols]
        cam_coord = (d_low[:, 0, ..., None, None] * torch.inverse(cam_mat[:, None, None]) @ pixels)[..., None, :, 0]
        ls_rdf = (N2C.unsqueeze(-3) @ self.ls).squeeze(-1) * self.rub_to_rdf
        if forward_mode == 'train':
            assert False, 'not implemented'
        elif forward_mode == 'test' or forward_mode == 'output':
            chunk = 10000
            bn, h, w, l, _ = ls_rdf.shape
            ls_rdf = ls_rdf.reshape([bn, 1, h * w, l, 3])
            cam_coord = cam_coord.reshape([bn, 1, h * w, 1, 3]).repeat(1, 1, 1, l, 1)
            x_list = []
            for j in range(0, h * w, chunk):
                cam_coord_j = cam_coord[:, :, j:j + chunk]
                ls_rdf_j = ls_rdf[:, :, j:j + chunk]
                xj = envmapfromVSG(vsg, cam_coord_j, ls_rdf_j, self.r_dist, data['bb'], cfg.sg_order)
                x_list.append(xj)

            pred_env_vsg = torch.cat(x_list, dim=2).reshape([bn, h, w, l, 3])
            env_constant_scale = 1000.0
            ls_rub = (ls_rdf * self.rub_to_rdf).reshape([bn, h, w, l, 3])
            diffuse, specular, _ = pbr(viewdir, ls_rub, n_low, a_low.permute([0, 2, 3, 1]).unsqueeze(-2),
                                       r_low.permute([0, 2, 3, 1]).unsqueeze(-2), self.ndotl, self.envWeight_ndotl,
                                       pred_env_vsg * env_constant_scale)

            diffuse = self.re_arr(diffuse)
            specular = self.re_arr(specular)
            diffscaled, specscaled, _, _ = LSregressDiffSpec(diffuse, specular, self.re_arr(all_rgb),
                                                             diffuse, specular, mask, scale_type=1)

            if forward_mode == 'test':
                assert False, 'not implemented'
                pred['e'] = pred_env_vsg
                pred['rgb'] = torch.clamp(diffscaled + specscaled, 0, 1.0)
                gt['rgb'] = self.re_arr(all_rgb)
                mask_dict = get_mask_dict('default', mask)
                mask_dict.update(get_mask_dict('env', mask[:, 0, ..., None, None]))
                return pred, gt, mask_dict

            if forward_mode == 'output':
                cDiff = (torch.sum(diffscaled) / torch.sum(diffuse)).data.item()
                cSpec = (torch.sum(specscaled)) / (torch.sum(specular)).data.item()
                if cSpec < 1e-3:
                    cAlbedo = 1 / a.max()
                    cLight = cDiff / cAlbedo
                else:
                    cLight = cSpec
                    cAlbedo = cDiff / cLight
                    cAlbedo = torch.clamp(cAlbedo, 1e-3, 1 / a.max())
                    cLight = cDiff / cAlbedo

                pred['vsg'] = (vsg, cLight.item())
                pred['n'], pred['d'], pred['a'], pred['r'] = n, d, a * cAlbedo, r
                pred['diff_vsg'], pred['spec_vsg'] = diffscaled, specscaled
                env = pred_env_vsg * cLight * env_constant_scale
                pred['e_vsg'] = rearrange(env, 'b r c (h w) C -> b C r c h w', h=env_h, w=env_w)
                pred['rgb_vsg'] = diffscaled + specscaled
                return pred, gt


@torch.no_grad()
def output_model_current(model, loader, gpu, cfg):
    if cfg.mode == 'MG':
        assert False, "Not implemented"
    else:
        for i, data in enumerate(tqdm(loader)):
            # names = data['name']
            data = tocuda(data, gpu, False)
            pred, gt = model.forward(data, cfg, 'output')
            outname = data['outname'][0]

            np.save(osp.join(outname, 'cam_hwf.npy'), data['hwf'][0].cpu().numpy())
            np.savez_compressed(osp.join(outname, 'vsg'),
                                vsg=np.ascontiguousarray(pred['vsg'][0].float().data.cpu().numpy()),
                                scale=pred['vsg'][1])

            saveImage(osp.join(outname, 'albedo.png'), pred['a'], is_hdr=False)  # is_hdr: do rgb_to_srgb conversion
            saveImage(osp.join(outname, 'normal.png'), 0.5 * (pred['n'] + 1), is_hdr=False)
            saveImage(osp.join(outname, 'rough.png'), pred['r'], is_hdr=False, is_single=True)
            saveImage(osp.join(outname, 'depth.png'), pred['d'], is_hdr=False, is_single=True)
            saveImage(osp.join(outname, 'color_gt.png'), data['i'], is_hdr=True)
            saveImage(osp.join(outname, 'color_pred.png'), pred['rgb_vsg'], is_hdr=True)
            saveImage(osp.join(outname, 'diffScaled.png'), pred['diff_vsg'], is_hdr=False)
            saveImage(osp.join(outname, 'specScaled.png'), pred['spec_vsg'], is_hdr=False)
            assert pred['e_vsg'].shape[0] == 1, f"batch size is not 1, {pred['e_vsg'].shape}"
            envmapsPredImage = pred['e_vsg'][0].float().data.cpu().numpy()
            envmapsPredImage = envmapsPredImage.transpose([1, 2, 3, 4, 0])  # img_h, img_w, env_h, env_w, 3
            np.savez_compressed(osp.join(outname, 'env'),
                                env=np.ascontiguousarray(envmapsPredImage[:, :, :, :, ::-1]))
            # writeEnvToFile(pred['e_vsg'].float(), 0, osp.join(outname, 'envmaps.png'), nrows=12, ncols=8)
            vis_env_map = env_util.visualize_pixelwise_env_maps(pred['e_vsg'].float()[0], nrows=24, ncols=16, gap=1,
                                                                rescale=True,
                                                                rgb2srgb=False,
                                                                output_type="numpy")
            vis_env_map = (vis_env_map.clip(min=0.0, max=1.0) * 255).astype(np.uint8)
            cv2.imwrite(osp.join(outname, 'envmaps.png'), vis_env_map[:, :, ::-1])

            if cfg.version == 'MAIR++':
                assert False, "Not implemented"
                saveImage(osp.join(outname, 'albedo_single.png'), pred['a_s'], is_hdr=True)
                saveImage(osp.join(outname, 'rough_single.png'), pred['r_s'], is_hdr=False, is_single=True)
                saveImage(osp.join(outname, 'color_pred_ilr.png'), pred['rgb_ilr'], is_hdr=True)
                saveImage(osp.join(outname, 'diffScaled_ilr.png'), pred['diff_ilr'], is_hdr=True)
                saveImage(osp.join(outname, 'specScaled_ilr.png'), pred['spec_ilr'], is_hdr=True)

                np.savez_compressed(osp.join(outname, 'ilr'),
                                    ilr=np.ascontiguousarray(pred['ilr'][0].float().data.cpu().numpy()))
                envmapsPredImage = pred['e_ilr'][0].float().data.cpu().numpy()
                envmapsPredImage = envmapsPredImage.transpose([1, 2, 3, 4, 0])
                np.savez_compressed(osp.join(outname, 'env_ilr'),
                                    env=np.ascontiguousarray(envmapsPredImage[:, :, :, :, ::-1]))
                writeEnvToFile(pred['e_ilr'].float(), 0, osp.join(outname, 'envmaps_ilr.png'), nrows=12, ncols=8)


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
        elif phase == 'custom_example_single':
            # test the images in Examples/ in a single view manner
            dataset = single_view_dataset.realworld_FF_single_view(dataRoot, cfg, outputRoot=outputRoot)
            is_shuffle = False
            print(f"Use realworld_FF_single_view, length: {dataset.length}")
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
    assert isinstance(model, MAIR.MAIRorg), f"model type: {type(model)}, expected: MAIR.MAIRorg"
    model.forward = MAIR_new_forward.__get__(model)  # bind the method to the model instance
    model.switch_to_eval()
    if dict_loaders is not None:
        for phase in dict_loaders:
            data_loader, _ = dict_loaders[phase]
            if run_mode == 'output':
                output_model_current(model, data_loader, gpu, cfg)
                # output_model(model, data_loader, gpu, cfg)

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


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process input data and configurations.")
    parser.add_argument('--dataroot', type=str, default='./Examples/input_processed', help='Path to the input data directory')
    parser.add_argument('--pretrained', type=str, default='pretrained/MAIR', help='Path to the pretrained model')
    parser.add_argument('--output_root', type=str, default='./out/03_single_view_02_adaptive_depth_range', help='Path to the output directory')
    parser.add_argument('--run_id', type=str, default='05190941_VSG', help='Identifier for the run')
    parser.add_argument('--run_mode', type=str, default='output', help='Mode of operation (e.g., output)')
    parser.add_argument('--phase_list', type=str, nargs='+', default=['custom'], help='List of phases to process')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    print(f"Arguments: {args}")
    test(0, 1, args.run_mode, args.phase_list, args.dataroot, args.pretrained, args.output_root,
         False, run_id=args.run_id, config=None)
