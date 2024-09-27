import os
import glob
import os.path as osp

from torch.utils.data import Dataset

from utils import *


class realworld_FF(Dataset):
    def __init__(self, dataRoot, cfg, img_w=320, img_h=240, outputRoot=None):
        self.img_w = img_w
        self.img_h = img_h
        self.cfg = cfg
        self.d_type = cfg.d_type
        self.env_size = (160, 120)
        self.size = (img_w, img_h)

        # colmap depth is so big compared to openrooms(meter)
        self.max_depth_type = 'pose'
        self.depth_max_scale = 10.0
        print(self.max_depth_type, self.depth_max_scale,
              'this must be same with realworld_FF_singleview(netdepth) value! ')

        sceneList = sorted(glob.glob(osp.join(dataRoot, '*')))
        if outputRoot is None:
            outroot = osp.join(osp.dirname(dataRoot), f'output/{cfg.version}')
        else:
            outroot = osp.join(outputRoot, cfg.version)
        print(f"read scene from {dataRoot}, {len(sceneList)} in total: {sceneList}")
        print(f"output to {outroot}")

        tmp = []
        index_to_remove = []  #
        for i in range(len(sceneList)):
            if 'main_xml' in sceneList[i] or sceneList[i].endswith('oi_only'):
                tmp += sorted(glob.glob(osp.join(sceneList[i], '*')))
                index_to_remove.append(i)
        for index in reversed(index_to_remove):
            del sceneList[index]
        sceneList += tmp

        all_idx = []
        for i in range(9):
            all_idx.append(str(i + 1))
        all_idx.remove('5')

        self.nameList = []
        self.idx_list = []
        self.is_real = []
        self.outname = []

        self.xy_offset = 1.3
        x, y, z = np.meshgrid(np.arange(self.cfg.VSGEncoder.vsg_res),
                              np.arange(self.cfg.VSGEncoder.vsg_res),
                              np.arange(self.cfg.VSGEncoder.vsg_res // 2), indexing='xy')
        x = x.astype(dtype=np.float32) + 0.5  # add half pixel
        y = y.astype(dtype=np.float32) + 0.5
        z = z.astype(dtype=np.float32) + 0.5
        z = z / (self.cfg.VSGEncoder.vsg_res // 2)
        x = self.xy_offset * (2.0 * x / self.cfg.VSGEncoder.vsg_res - 1)
        y = self.xy_offset * (2.0 * y / self.cfg.VSGEncoder.vsg_res - 1)
        self.voxel_grid = [x, y, z]

        self.hdr_postfix = 'rgbe'
        for j, scene in enumerate(sceneList):
            if osp.exists(osp.join(scene, 'pair.txt')):
                # continue
                pair_file = 'pair.txt'
                # pair_file = osp.join(scene, 'pair.txt')
                with open(osp.join(scene, pair_file), 'r') as f:
                    num_viewpoint = int(f.readline().strip())
                    # viewpoints
                    for view_idx in range(num_viewpoint):
                        ref_view = int(f.readline().rstrip())
                        src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                        if ref_view % 4 != 1:
                            continue
                        if len(src_views) == 0:
                            continue
                        # filter by no src view and fill to nviews
                        src_views = src_views[:8]

                        outfilename_org = osp.join(outroot, osp.basename(scene))
                        outfilename = f'{outfilename_org}_{(ref_view + 1):03d}'
                        os.makedirs(outfilename, exist_ok=True)
                        if cfg.version == 'MAIR++':
                            if len(os.listdir(outfilename)) == 20:
                                continue
                        if cfg.version == 'MAIR':
                            if len(os.listdir(outfilename)) == 12:
                                continue
                        self.nameList.append(scene + '$' + str(ref_view + 1))
                        self.idx_list.append(list(map(lambda x: str(x + 1), src_views)))
                        self.is_real.append(True)
                        self.outname.append(outfilename)

            else:
                a = sorted(list(set([b.split('_')[0] for b in os.listdir(scene)])))
                for t in a:
                    outfilename_org = osp.join(outroot, osp.basename(osp.dirname(scene)) + '_' + osp.basename(scene))
                    outfilename = f'{outfilename_org}_{int(t):03d}'
                    os.makedirs(outfilename, exist_ok=True)
                    if cfg.version == 'MAIR++':
                        if len(os.listdir(outfilename)) == 20:
                            continue
                    if cfg.version == 'MAIR':
                        if len(os.listdir(outfilename)) == 12:
                            continue
                    self.nameList.append(scene + '$' + t)
                    self.idx_list.append(all_idx)
                    self.is_real.append(False)
                    self.outname.append(outfilename)
        self.length = len(self.nameList)

    def __len__(self):
        return self.length

    def __getitem__(self, ind):
        batch = {}
        training_idx = self.idx_list[ind].copy()
        is_real = self.is_real[ind]
        batch['outname'] = self.outname[ind]
        if is_real:
            scene, target_idx = self.nameList[ind].split('$')
            all_idx = [target_idx, ] + training_idx
            name_list = [osp.join(scene, 'images_320x240', '{}_' + f'{int(a):03d}' + '.{}') for a in all_idx]
            cam_name = osp.join(scene, 'images_320x240/cam_mats.npy')

            im = cv2.imread(name_list[0].format('im', 'png'), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            im = im[..., ::-1].astype(np.float32) / 255.0
            im = ldr2hdr(im).transpose([2, 0, 1])

            cam_mats = np.load(cam_name)
        else:
            scene, scene_idx = self.nameList[ind].split('$')
            target_idx = '5'
            all_idx = [target_idx, ] + training_idx
            assert training_idx == ['1', '2', '3', '4', '6', '7', '8', '9']
            name_list = [osp.join(scene, scene_idx + '_{}_' + a + '.{}') for a in all_idx]
            cam_name = osp.join(scene, f'{scene_idx}_cam_mats.npy')

            seg_name = name_list[0].format('immask', 'png')
            seg_large = (loadImage(seg_name, type='s'))[..., :1]
            im = loadImage(name_list[0].format('im', self.hdr_postfix), 'i')
            scale = get_hdr_scale(im, seg_large > 0.9, 'test')
            im = cv2.resize(im * scale, self.size, interpolation=cv2.INTER_AREA)
            im = np.clip(im, 0, 1.0).transpose([2, 0, 1])

            cam_mats = np.load(cam_name)
            h, w, f = cam_mats[:, 4, 0]
            cam_mats[:, 4, :] = cam_mats[:, 4, :] / (w / self.size[0])

        batch['i'] = im
        batch['m'] = np.ones_like(im[:1])

        cds_conf_name = name_list[0].format('cdsconf', 'dat')
        cds_conf = loadImage(cds_conf_name, 'd', self.size, normalize=False).transpose([2, 0, 1])
        batch['cds_conf'] = cds_conf

        if self.max_depth_type == 'pose':
            # mvsd_pose : for openrooms or for oi and real-world
            max_depth = cam_mats[1, -1, int(target_idx) - 1].astype(np.float32)
        elif self.max_depth_type == 'est':
            # mvsd_est : for ir and real-world
            target_conf = loadBinary(name_list[0].format('cdsconf', 'dat'))
            target_conf = target_conf > 0.6
            target_depth = loadBinary(name_list[0].format('cdsdepthest', 'dat'))
            max_depth = np.max(target_conf * target_depth)

        cds_depth_name = name_list[0].format('cdsdepthest', 'dat')
        cds_depth = loadImage(cds_depth_name, 'd', self.size, normalize=False).transpose([2, 0, 1])
        batch['cds_dn'] = np.clip(cds_depth / max_depth, 0, 1)
        grad_x = cv2.Sobel(batch['cds_dn'][0], -1, 1, 0)
        grad_y = cv2.Sobel(batch['cds_dn'][0], -1, 0, 1)
        batch['cds_dg'] = cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)[None]

        poses_hwf_bounds = cam_mats[..., int(target_idx) - 1]
        h, w, f = poses_hwf_bounds[:, -2]
        intrinsic = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=float).astype(np.float32)
        batch['cam'] = intrinsic
        batch['hwf'] = np.array([h, w, f])

        if hasattr(self, 'voxel_grid'):
            fov_x = intrinsic[0, 2] / intrinsic[0, 0]
            fov_y = intrinsic[1, 2] / intrinsic[0, 0]
            batch['bb'] = np.array([self.xy_offset * fov_x, self.xy_offset * fov_y, 1.05], dtype=np.float32)
            x = self.voxel_grid[0] * fov_x
            y = self.voxel_grid[1] * fov_y
            z = self.voxel_grid[2] * 1.05
            batch['voxel_grid_front'] = np.stack([x, y, z], axis=-1)

        depth_scale = 1.0
        if is_real:
            # if scene's max depth is larger than depth_max_scale, we scale down depth.
            depth_scale = max(1.0, max_depth / self.depth_max_scale)
        cam_mats[:, 3, :] /= depth_scale

        src_c2w_list = []
        src_int_list = []
        rgb_list = []
        depthest_list = []
        fac = self.env_size[1] / self.size[1]
        for name, idx in zip(name_list, all_idx):
            if is_real:
                im = cv2.imread(name.format('im', 'png'), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                im = cv2.resize(im, self.env_size, interpolation=cv2.INTER_AREA)
                im = ldr2hdr(im[..., ::-1].astype(np.float32) / 255.0)
            else:
                im = loadImage(name.format('im', self.hdr_postfix), 'i', self.env_size)
                im = np.clip(im * scale, 0, 1.0)
            rgb_list.append(im)

            poses_hwf_bounds = cam_mats[..., int(idx) - 1]
            src_c2w_list.append(np34_to_44(poses_hwf_bounds[:, :4]))
            cy2, cx2, fx = poses_hwf_bounds[:, -2]
            fy = poses_hwf_bounds[-1, -1]
            if fy == 0:
                fy = fx
            intrinsic = np.array([[fx * fac, 0, cx2 / 2 * fac], [0, fy * fac, cy2 / 2 * fac], [0, 0, 1]], dtype=float)
            src_int_list.append(intrinsic)
            if self.d_type == 'cds':
                depth = loadImage(name.format('cdsdepthest', 'dat'), 'd', self.env_size, False)
                depth = depth / depth_scale
            elif self.d_type == 'net':
                # netdepth is already divided by depth scale.
                depth = loadImage(name.format('netdepth', 'dat'), 'd', self.env_size, False)
            depthest_list.append(depth)

        batch['all_i'] = np.stack(rgb_list, axis=0).transpose([0, 3, 1, 2])
        batch['all_cam'] = np.stack(src_int_list, axis=0).astype(np.float32)
        w2target = np.linalg.inv(src_c2w_list[0])
        batch['c2w'] = (w2target @ np.stack(src_c2w_list, 0)).astype(np.float32)
        batch['all_depth'] = np.stack(depthest_list, axis=0).transpose([0, 3, 1, 2])
        return batch