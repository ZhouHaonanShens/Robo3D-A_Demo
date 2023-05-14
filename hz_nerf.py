"""
@Project : Robo3D
@File    : hz_nerf.py
@IDE     : PyCharm
@Authors : Haonan Zhou
@Date    : 10/05/2023
@Brief   : The main/entry script to run the nerf training or testing.
@Parameters are all explained in its own 'help'
"""

"""
# Parameters:
@path           :   (str or path)The path to data---No Default Value
@workspace      :   (str or path)The path to workspace for logs and results---default='workspace'
@ckpt           :   (str or path)The path to the checkpoint---default='latest'
@cuda_ray       :   (store true)Use CUDA raymarching instead of pytorch if ture---default=False
@bound          :   (float)assume the scene is bounded in box[-bound, bound]^3,
                    if > 1, will invoke adaptive ray marching.'---default=2
@scale          :   (float)Scale camera location into box[-bound, bound]^3---default=0.33
@max_ray_batch  :   (int)batch size of rays at inference to avoid OOM, 
                    only valid when NOT using '--cuda_ray'---default=4096
@depth_supervise:   (store true)Introduce depth supervision via KL divergence---default=False
@test_entropy   :   (store true)Enable entropy test only mode---default=False
@hz_train       :   (store true)Enable the active selecting policy---default=False
"""

import numpy as np
import torch
import argparse

from nerf.provider import NeRFDataset
from nerf.gui import NeRFGUI
from nerf.utils import *

from functools import partial
from loss import huber_loss

# torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('-O', action='store_true', help="equals --fp16 --cuda_ray --preload")
    parser.add_argument('-test_entropy', action='store_true', help="Enable Entropy Test")
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=0)

    ### training options
    parser.add_argument('--iters', type=int, default=159 * 50 + 1, help="training iters")
    parser.add_argument('--lr', type=float, default=1e-2, help="initial learning rate")
    parser.add_argument('--ckpt', type=str, default='latest')
    parser.add_argument('--num_rays', type=int, default=4096, help="num rays sampled per image for each training step")
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--max_steps', type=int, default=1024,
                        help="max num steps sampled per ray (only valid when using --cuda_ray)")
    parser.add_argument('--num_steps', type=int, default=512,
                        help="num steps sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--upsample_steps', type=int, default=0,
                        help="num steps up-sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--update_extra_interval', type=int, default=16,
                        help="iter interval to update extra status (only valid when using --cuda_ray)")
    parser.add_argument('--max_ray_batch', type=int, default=4096,
                        help="batch size of rays at inference to avoid OOM (only valid when NOT using --cuda_ray)")
    parser.add_argument('--patch_size', type=int, default=1,
                        help="[experimental] render patches in training, so as to apply LPIPS loss. 1 means disabled, use [64, 32, 16] to enable")

    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")

    ### dataset options
    parser.add_argument('--color_space', type=str, default='srgb', help="Color space, supports (linear, srgb)")
    parser.add_argument('--preload', action='store_true',
                        help="preload all data into GPU, accelerate training but use more GPU memory")
    # (the default value is for the fox dataset)
    parser.add_argument('--bound', type=float, default=2,
                        help="assume the scene is bounded in box[-bound, bound]^3, if > 1, will invoke adaptive ray marching.")
    parser.add_argument('--scale', type=float, default=0.33, help="scale camera location into box[-bound, bound]^3")
    parser.add_argument('--offset', type=float, nargs='*', default=[0, 0, 0], help="offset of camera location")
    parser.add_argument('--dt_gamma', type=float, default=1 / 128,
                        help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--min_near', type=float, default=0.2, help="minimum near distance for camera")
    parser.add_argument('--density_thresh', type=float, default=10, help="threshold for density grid to be occupied")
    parser.add_argument('--bg_radius', type=float, default=1,
                        help="if positive, use a background model at sphere(bg_radius)")

    ### GUI options
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=1920, help="GUI width")
    parser.add_argument('--H', type=int, default=1080, help="GUI height")
    parser.add_argument('--radius', type=float, default=1, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=50, help="default GUI camera fovy")
    parser.add_argument('--max_spp', type=int, default=64, help="GUI rendering max sample per pixel")

    ### experimental
    parser.add_argument('--error_map', action='store_true', help="use error map to sample rays")
    parser.add_argument('--clip_text', type=str, default='', help="text input for CLIP guidance")
    parser.add_argument('--rand_pose', type=int, default=-1,
                        help="<0 uses no rand pose, =0 only uses rand pose, >0 sample one rand pose every $ known poses")

    parser.add_argument('--HZ', action='store_true', help="Cancel --cuda_ray and enable entropy")
    parser.add_argument('--downscale', type=int, default=8, help="downscale to avoid lack of GPU memory")
    parser.add_argument('--depth_supervise', action='store_true', help="Introduce depth supervision via KL divergence")
    parser.add_argument('--depth_supervise_E2', action='store_true', help="Introduce depth supervision via KL divergence")
    # parser.add_argument('--test_entropy', action='store_true')
    parser.add_argument('--hz_train', action='store_true', help="Train with new policy.")

    opt = parser.parse_args()

    if opt.O:
        opt.fp16 = True
        opt.cuda_ray = True
        opt.preload = True
        opt.test_entropy = False

    if opt.HZ:
        opt.fp16 = True
        opt.preload = True
        opt.cuda_ray = True
        # opt.test_entropy = False

    test_entropy = opt.test_entropy
    if opt.depth_supervise:
        opt.cuda_ray = False

    if opt.patch_size > 1:
        opt.error_map = False  # do not use error_map, if use patch-based training
        # assert opt.patch_size > 16, "patch_size should > 16 to run LPIPS loss."
        assert opt.num_rays % (opt.patch_size ** 2) == 0, "patch_size ** 2 should be dividable by num_rays."

    from nerf.network import NeRFNetwork

    print(opt)

    seed_everything(opt.seed)

    model = NeRFNetwork(
        encoding="hashgrid",
        bound=opt.bound,
        cuda_ray=opt.cuda_ray,
        density_scale=1,
        min_near=opt.min_near,
        density_thresh=opt.density_thresh,
        bg_radius=opt.bg_radius,
    )

    print(model)

    criterion = torch.nn.MSELoss(reduction='none')
    # criterion = partial(huber_loss, reduction='none')
    # criterion = torch.nn.HuberLoss(reduction='none', beta=0.1) # only available after torch 1.10 ?

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if opt.test:

        metrics = [PSNRMeter(), LPIPSMeter(device=device)]
        trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, criterion=criterion, fp16=opt.fp16,
                          metrics=metrics, use_checkpoint=opt.ckpt)

        if opt.gui:
            gui = NeRFGUI(opt, trainer)
            gui.render()

        else:
            test_loader = NeRFDataset(opt, device=device, type='test').dataloader()

            if test_loader.has_gt:
                trainer.evaluate(test_loader)  # blender has gt, so evaluate it.

            trainer.test(test_loader, write_video=True)  # test and save video

            trainer.save_mesh(resolution=256, threshold=10)

    elif opt.test_entropy:
        metrics = [PSNRMeter(), LPIPSMeter(device=device)]
        trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, criterion=criterion, fp16=opt.fp16,
                          metrics=metrics, use_checkpoint=opt.ckpt)
        max_idxs = []
        max_values = []
        for zone_idx in range(12):
            type_zone = f"zone{zone_idx:02d}"
            test_loader = NeRFDataset(opt, device=device, type=type_zone, downscale=8).dataloader()
            entropy_means = []
            for i, data in enumerate(test_loader):
                w_fine_entropy = trainer.entropy_step(data)
                entropy_mean = np.mean(w_fine_entropy)
                entropy_means.append(entropy_mean)
            max_idx = np.argmax(np.array(entropy_means))
            max_value = np.max(np.array(entropy_means))
            max_idxs.append(max_idx)
            max_values.append(max_value)
        print('max_idxs:', max_idxs)
        print('max_value:', max_values)

    elif opt.hz_train:
        optimizer = lambda model: torch.optim.Adam(model.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15)

        # train_loader = NeRFDataset(opt, device=device, type='train').dataloader()

        # decay to 0.1 * init_lr at last iter step
        scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer,
                                                                  lambda iter: 0.1 ** min(iter / opt.iters, 1))

        metrics = [PSNRMeter(), LPIPSMeter(device=device)]
        trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, optimizer=optimizer,
                          criterion=criterion, ema_decay=0.95, fp16=opt.fp16, lr_scheduler=scheduler,
                          scheduler_update_every_step=True, metrics=metrics, use_checkpoint=opt.ckpt, eval_interval=50)

        # valid_loader = NeRFDataset(opt, device=device, type='val', downscale=1).dataloader()

        # max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)
        max_epoch = 500
        train_set = NeRFDataset(opt, device=device, type='train')
        val_set = NeRFDataset(opt, device=device, type='val')
        trainer.hz_train(train_set, val_set, max_epoch)

    else:
        optimizer = lambda model: torch.optim.Adam(model.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15)

        train_loader = NeRFDataset(opt, device=device, type='train').dataloader()

        # decay to 0.1 * init_lr at last iter step
        scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer,
                                                                  lambda iter: 0.1 ** min(iter / opt.iters, 1))

        metrics = [PSNRMeter(), LPIPSMeter(device=device)]
        trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, optimizer=optimizer,
                          criterion=criterion, ema_decay=0.95, fp16=opt.fp16, lr_scheduler=scheduler,
                          scheduler_update_every_step=True, metrics=metrics, use_checkpoint=opt.ckpt, eval_interval=50)

        if opt.gui:
            gui = NeRFGUI(opt, trainer, train_loader)
            gui.render()

        else:
            valid_loader = NeRFDataset(opt, device=device, type='val', downscale=1).dataloader()

            max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)
            trainer.train(train_loader, valid_loader, max_epoch)

            # also test
            test_loader = NeRFDataset(opt, device=device, type='test').dataloader()

            if test_loader.has_gt:
                trainer.evaluate(test_loader)  # blender has gt, so evaluate it.

            trainer.test(test_loader, write_video=True)  # test and save video

            trainer.save_mesh(resolution=256, threshold=10)
