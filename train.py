import datetime
import os

import torch
from torch import nn
from torch import optim
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import joint_transforms
from config import ViSha_training_root, ViSha_test_root
from dataset.VSshadow_ours import CrossPairwiseImg 
from misc import AvgMeter, check_mkdir
import math
from losses import lovasz_hinge, binary_xloss, structure_loss
import random
import torch.nn.functional as F
import numpy as np
import pdb 
from torchvision.utils import save_image, make_grid
import time
import argparse
import importlib
from utils import backup_code
from torch.utils.tensorboard import SummaryWriter
import os 

from loss.monodepth_loss import MonodepthLoss

os.CUDA_VISIBLE_DEVICES = '0'

cudnn.deterministic = True
cudnn.benchmark = False

ckpt_path = 'models'

# VMD_file = importlib.import_module('networks.' + model_name)
from networks.VGD_reflection import VGD_Network


_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


def freeze_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def denorm(img_tensor):
    return img_tensor * torch.tensor(_STD).view(3, 1, 1) + torch.tensor(_MEAN).view(3, 1, 1)


def freeze_other_parameters(net, list):
    for name, param in net.named_parameters():
        if not any(fine_tuning_name in name for fine_tuning_name in list):
            param.requires_grad = False
            # print(name, 'is frozen')
        else:
            print(name, 'is not frozen')
            # pass 

def main(cmd_args):
    exp_name = cmd_args.exp
    model_name = cmd_args.model
    gpu_ids = cmd_args.gpu
    train_batch_size = cmd_args.batchsize

    print('='*10)
    print(cmd_args)
    print('='*10, '\n\n')

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids

    args = {
        # 'exp_name': exp_name,
        'max_epoch': 15,
        # 'train_batch_size': 10,
        'last_iter': 0,

        'finetune_lr': cmd_args.finetune_lr, 
        'scratch_lr': cmd_args.scratch_lr,

        'weight_decay': 5e-4,
        'momentum': 0.9,
        'snapshot': '',
        'scale': cmd_args.scale,
        'multi-scale': None,
        # 'gpu': '4,5',
        # 'multi-GPUs': True,
        'fp16': False,
        'warm_up_epochs': 1,  #### NOTE: default is 3
        'seed': 2023,
        'monodepth_lambda': 0.1
    }
    # fix random seed
    np.random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    torch.cuda.manual_seed(args['seed'])

    # multi-GPUs training
    if len(gpu_ids.split(',')) > 1:
        batch_size = train_batch_size * len(gpu_ids.split(','))
    # single-GPU training
    else:
        torch.cuda.set_device(0)
        batch_size = train_batch_size

    print('batch_size: {}'.format(batch_size))
    print('batch_size: {}'.format(batch_size))
    print(args)


    print('=====>Dataset loading<======')
    training_root = [ViSha_training_root] # training_root should be a list form, like [datasetA, datasetB, datasetC], here we use only one dataset.
    train_set = CrossPairwiseImg(training_root)
    # train_subset = torch.utils.data.Subset(train_set, range(100))
    # train_loader_subset = DataLoader(train_subset, batch_size=batch_size, num_workers=1, shuffle=False)
    train_loader = DataLoader(train_set, ##### NOTE: more training data!!!!
                            batch_size=batch_size,  drop_last=True, num_workers=cmd_args.num_workers,  
                            shuffle=True)


    val_set = CrossPairwiseImg([ViSha_test_root])
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=cmd_args.num_workers, shuffle=True)   ## shuffle for better visualization


    print("max epoch:{}".format(args['max_epoch']))

    ce_loss = nn.CrossEntropyLoss()

    log_path = os.path.join(ckpt_path, exp_name, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '.txt')
    val_log_path = os.path.join(ckpt_path, exp_name, 'val_log' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '.txt')
    writer = SummaryWriter(log_dir=os.path.join(ckpt_path, exp_name, 'board'))

    print('=====>Prepare Network {}<======'.format(exp_name))
    # multi-GPUs training
    if len(gpu_ids.split(',')) > 1:
        net = torch.nn.DataParallel(VGD_Network(height=args['scale'], width=args['scale'])).cuda().train()
        model_without_ddp = net.module 
        # for name, param in net.named_parameters():
        #     if 'backbone' in name:
        #         print(name)
        print('Multi-GPU training, using {} GPUs'.format(len(gpu_ids.split(','))))
        params = [
            {"params": [param for name, param in net.named_parameters() if 'backbone' in name], 
             "lr": args['finetune_lr']},
            {"params": [param for name, param in net.named_parameters() if 'backbone' not in name], 
             "lr": args['scratch_lr']},
        ]
    # single-GPU training
    else:
        net = VGD_Network(height=args['scale'], width=args['scale']).cuda().train()
        ## net = net.apply(freeze_bn) # freeze BN
        model_without_ddp = net 
        params = [
            {"params": [param for name, param in net.named_parameters() if 'backbone' in name], 
             "lr": args['finetune_lr']},
            {"params": [param for name, param in net.named_parameters() if 'backbone' not in name], 
             "lr": args['scratch_lr']},
        ]
    
    print("Number of parameters: ", sum(p.numel() for p in net.parameters() if p.requires_grad))

    # optimizer = optim.SGD(params, momentum=args['momentum'], weight_decay=args['weight_decay'])
    optimizer = optim.Adam(params, betas=(0.9, 0.99), eps=6e-8, weight_decay=args['weight_decay'])
    warm_up_with_cosine_lr = lambda epoch: epoch / args['warm_up_epochs'] if epoch <= args['warm_up_epochs'] and args['warm_up_epochs'] > 0 else 0.5 * \
                             (math.cos((epoch-args['warm_up_epochs'])/(args['max_epoch']-args['warm_up_epochs'])*math.pi)+1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)
    # scheduler = StepLR(optimizer, step_size=3, gamma=0.1)  # change learning rate after 20000 iters    

    scaler = torch.GradScaler('cuda', enabled = args['fp16'])

    if cmd_args.resume is not None:
        print('=====>Loading checkpoint {}<======'.format(cmd_args.resume))
        checkpoint = torch.load(cmd_args.resume)
        msg = model_without_ddp.load_state_dict(checkpoint['model'], strict=False) ###
        print(msg)
        print(optimizer, checkpoint['optimizer'], '====')
        optimizer.load_state_dict(checkpoint['optimizer'])  
        scheduler.load_state_dict(checkpoint['scheduler'])
        curr_epoch = checkpoint['curr_epoch'] + 1  ###
        print('=====>Loaded checkpoint {}<======'.format(cmd_args.resume))
    else:
        curr_epoch = 1 
    
    check_mkdir(ckpt_path)
    check_mkdir(os.path.join(ckpt_path, exp_name))
    backup_code(".", os.path.join(ckpt_path, exp_name, "backup_code"))
    with open(os.path.join(os.path.dirname(__file__), log_path), 'w') as f:
        f.write(str(args) + '\n\n')
        f.write(str(cmd_args) + '\n')
        f.write(str(optimizer) + '\n\n')

    # Training
    curr_iter = 1
    start = 0
    best_mae = 100.0

    # current_mae = val(net, curr_epoch) 
    # net.train()

    print('=====>Start training<======')
    while True:
        loss_record1, loss_record2, loss_record3, loss_record4 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
        loss_record5, loss_record6, loss_record7, loss_record8, loss_record9 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
        loss_record10, loss_record11, loss_record12, loss_record13, loss_record14 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
        mono_loss_record = AvgMeter()
        train_iterator = tqdm(train_loader, total=len(train_loader))
        # train_iterator = tqdm(train_loader, desc=f'Epoch: {curr_epoch}', ncols=100, ascii=' =', bar_format='{l_bar}{bar}|')
        # tqdm(train_loader, total=len(train_loader))

        monodepth_options = {
            "frame_ids": [0,1,-1],
            "num_scales": 4,
            "height": args["scale"],
            "width": args["scale"],
            "crop_h": args["scale"],
            "crop_w": args["scale"]
        }


        monodepth_loss_args = {
            "min_depth": 0.1,
            "max_depth": 100,
            "test_min_depth": 1.0e-3,
            "test_max_depth": 80,
            "disparity_smoothness": 1.0e-3,
            "no_ssim": False,
            "avg_reprojection": False,
            "disable_automasking": False
            }
        
        monodepth_loss_args.update(monodepth_options)

        monodepth_loss_calculator_train = MonodepthLoss(**monodepth_loss_args, batch_size=batch_size, is_train=True)
        monodepth_loss_calculator_val = MonodepthLoss(**monodepth_loss_args, batch_size=batch_size, is_train=False)

        
        for i, sample in enumerate(train_iterator):
            for k, v in sample.items():
                if torch.is_tensor(v):
                    sample[k] = v.cuda()

            exemplar, exemplar_gt, query, query_gt = sample[("color_aug", 0, 0)], sample[("gt", 0)], sample[("color_aug", 1, 0)], sample[("gt", 1)]
            other, other_gt = sample[("color_aug", -1, 0)], sample[("gt", -1)]


            # exemplar_ref_gt, query_ref_gt, other_ref_gt = sample['exemplar_ref'].cuda(), sample['query_ref'].cuda(), sample['other_ref'].cuda()

            B = exemplar.size(0)

            optimizer.zero_grad()
            
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args['fp16']):
                exemplar_pre, query_pre, other_pre, \
                    exemplar_final, query_final, other_final, \
                        outputs_exp, outputs_query, outputs_other, \
                             poses = net(exemplar, query, other, sample)
            
            # #### x gt mask 
            # ref_loss1 = torch.sum(torch.mean(((exemplar_ref - exemplar_ref_gt)**2).view(B, -1), 1))
            # ref_loss2 = torch.sum(torch.mean(((query_ref - query_ref_gt)**2).view(B, -1), 1))
            # ref_loss3 = torch.sum(torch.mean(((other_ref - other_ref_gt)**2).view(B, -1), 1))
            # if cmd_args.loss_ref_penalty > 0:
            #     penalty1 = torch.sum(torch.mean(((exemplar_ref * exemplar_gt - exemplar_ref)**2).view(B, -1), 1))
            #     penalty2 = torch.sum(torch.mean(((query_ref * query_gt - query_ref)**2).view(B, -1), 1))
            #     penalty3 = torch.sum(torch.mean(((other_ref * other_gt - other_ref)**2).view(B, -1), 1))
            #     ref_loss1 = ref_loss1 + penalty1 * cmd_args.loss_ref_penalty 
            #     ref_loss2 = ref_loss2 + penalty2 * cmd_args.loss_ref_penalty 
            #     ref_loss3 = ref_loss3 + penalty3 * cmd_args.loss_ref_penalty  

            loss_hinge1 = structure_loss(exemplar_pre, exemplar_gt)  
            loss_hinge2 = structure_loss(query_pre, query_gt)
            loss_hinge3 = structure_loss(other_pre, other_gt)
            loss_hinge_examplar = structure_loss(exemplar_final, exemplar_gt)
            loss_hinge_query = structure_loss(query_final, query_gt)
            loss_hinge_other = structure_loss(other_final, other_gt) 
            
            loss_seg = loss_hinge1 + loss_hinge2 + loss_hinge3 + loss_hinge_examplar + loss_hinge_query + loss_hinge_other
            # loss_ref = ref_loss1 + ref_loss2 + ref_loss3

            loss = loss_seg #+ loss_ref 

            scaler.scale(loss).backward(retain_graph=True)
            
            outputs_exp.update(poses)
            if args['monodepth_lambda'] > 0:
                monodepth_loss_calculator_train.generate_images_pred(sample, outputs_exp)
                mono_losses = monodepth_loss_calculator_train.compute_losses(sample, outputs_exp)
                mono_loss = args['monodepth_lambda'] * mono_losses["loss"]

                scaler.scale(mono_loss).backward()

            torch.nn.utils.clip_grad_norm_(net.parameters(), 12)  # gradient clip
            scaler.step(optimizer)  # change gradient
            scaler.update()

            loss_record1.update(loss_hinge_examplar.item(), batch_size)
            loss_record2.update(loss_hinge_query.item(), batch_size)
            loss_record3.update(loss_hinge_other.item(), batch_size)
            loss_record4.update(loss_hinge1.item(), batch_size)
            loss_record5.update(loss_hinge2.item(), batch_size)
            loss_record6.update(loss_hinge3.item(), batch_size)
            mono_loss_record.update(mono_loss.item(), batch_size)
            # loss_record7.update(ref_loss1.item(), batch_size)
            # loss_record8.update(ref_loss2.item(), batch_size)
            # loss_record9.update(ref_loss3.item(), batch_size)


            writer.add_scalars('Loss/train_hinge', {'loss_hinge_examplar': loss_record1.avg, 'loss_hinge_query': loss_record2.avg, 'loss_hinge_other': loss_record3.avg}, curr_iter)
            writer.add_scalars('Loss/train_final', {'loss_hinge1': loss_record4.avg, 'loss_hinge2': loss_record5.avg, 'loss_hinge3': loss_record6.avg,}, curr_iter)
            writer.add_scalars('Loss/train_mono', {'mono_loss': mono_loss_record.avg}, curr_iter)
            #writer.add_scalars('Loss/train_ref', {'ref_loss1': loss_record7.avg, 'ref_loss2': loss_record8.avg, 'ref_loss3': loss_record9.avg,}, curr_iter)

            # if cmd_args.loss_ref_penalty > 0:
            #     writer.add_scalars('Loss/penalty', {'penalty1': penalty1.item(), 'penalty2': penalty2.item(), 'penalty3': penalty3.item()}, curr_iter)
            writer.add_scalars('lr', {'lr': scheduler.get_lr()[0]}, curr_iter)
            writer.add_scalars('lr', {'lr_group0': optimizer.param_groups[0]['lr']}, curr_iter)
            writer.add_scalars('lr', {'lr_group1': optimizer.param_groups[1]['lr']}, curr_iter)

            curr_iter += 1

            log = "epochs:%d, iter: %d, hinge1_f: %f5, hinge2_f: %f5, hinge3_f: %f5, hinge1: %f5, hinge2: %f5, hinge3: %f5, ref1: %f5, ref2: %f5, ref3: %f5, edge1: %f5, edge2: %f5, edge3: %f5,lr: %f8"%\
                  (curr_epoch, curr_iter, 
                   loss_record1.avg, loss_record2.avg, loss_record3.avg, 
                   loss_record4.avg, loss_record5.avg, loss_record6.avg, 
                   loss_record7.avg, loss_record8.avg, loss_record9.avg, 
                   loss_record10.avg, loss_record11.avg, loss_record12.avg, scheduler.get_lr()[0])
            
            log += f". group[0].lr: {optimizer.param_groups[0]['lr']:.8f}, group[1].lr: {optimizer.param_groups[1]['lr']:.8f}"

            if (curr_iter-1) % 20 == 0:
                elapsed = (time.perf_counter() - start)
                start = time.perf_counter()
                log_time = log + ' [time {}]'.format(elapsed)
                print(log_time)
                # train_iterator.set_description(log_time)
            with open(log_path, 'a') as file:
                file.write(log + '\n')

        if curr_epoch % 1 == 0 and not cmd_args.bestonly:
            # if args['multi-GPUs']:
            if len(gpu_ids.split(',')) > 1:
                if args['fp16']:
                    checkpoint = {
                        'model': net.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        #'amp': amp.state_dict(),
                        'curr_epoch': curr_epoch
                    }
                else:
                    checkpoint = {
                        'model': net.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'curr_epoch': curr_epoch
                    }
                torch.save(checkpoint, os.path.join(ckpt_path, exp_name, f'{curr_epoch}.pth'))
            else:
                # torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % curr_epoch))
                if args['fp16']:
                    checkpoint = {
                        'model': net.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        #'amp': amp.state_dict(),
                        'curr_epoch': curr_epoch
                    }
                else:
                    checkpoint = {
                        'model': net.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'curr_epoch': curr_epoch
                    }
                torch.save(checkpoint, os.path.join(ckpt_path, exp_name, f'{curr_epoch}.pth'))
            print('save model ...', os.path.join(ckpt_path, exp_name, f'{curr_epoch}.pth'))


        try:
            ### visualize
            vis_save_path = os.path.join(ckpt_path, exp_name, 'vis')
            os.makedirs(vis_save_path, exist_ok=True)
            save_image(denorm(exemplar[:B].cpu()), os.path.join(vis_save_path, f'{curr_epoch}.png'), nrow=B)
            #save_image(exemplar_ref[:B].cpu(), os.path.join(vis_save_path, f'{curr_epoch}_ref.png'), nrow=B)
            save_image(make_grid(torch.cat([exemplar_pre[:B], exemplar_gt[:B]], dim=0), nrow=B), 
                    os.path.join(vis_save_path, f'{curr_epoch}_medge.png'))
        except:
            print('Visualization error !!')

        try:
            current_mae = val(net, curr_epoch, val_loader, val_log_path, vis_save_path)
        except:
            current_mae = val(net, curr_epoch, val_loader, val_log_path)



        writer.add_scalars('Validation', {'mae': current_mae}, curr_epoch)

        net.train() # val -> train
        if len(gpu_ids.split(',')) > 1:
            if args['fp16']:
                checkpoint = {
                    'model': net.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'curr_epoch': curr_epoch,
                    #'amp': amp.state_dict()
                }
            else:
                checkpoint = {
                    'model': net.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'curr_epoch': curr_epoch,
                }
        else:
            if args['fp16']:
                checkpoint = {
                    'model': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    #'amp': amp.state_dict(),
                    'curr_epoch': curr_epoch,
                }
            else:
                checkpoint = {
                    'model': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'curr_epoch': curr_epoch,
                }
        if current_mae < best_mae:
            best_mae = current_mae
            torch.save(checkpoint, os.path.join(ckpt_path, exp_name, 'best_mae.pth'))
        torch.save(checkpoint, os.path.join(ckpt_path, exp_name, 'latest.pth'))

        if curr_epoch > args['max_epoch']:
            # torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % curr_iter))
            return
        curr_epoch += 1
        scheduler.step()  # change learning rate after epoch

def val(net, epoch, val_loader, val_log_path, vis_save_path=None):
    mae_record = AvgMeter()
    net.eval()
    with torch.no_grad():
        val_iterator = tqdm(val_loader)
        for i, sample in enumerate(val_iterator):
            for k, v in sample.items():
                if torch.is_tensor(v):
                    sample[k] = v.cuda()


            exemplar, query, other = sample[("color_full", 0, 0)], sample[("color_full", 1, 0)], sample[("color_full", -1, 0)]
            exemplar_gt = sample[("gt_resized", 0)]

            B = exemplar_gt.shape[0]

            examplar_final = net(exemplar, query, other, sample)

            res = (examplar_final.data > 0).to(torch.float32).squeeze(0)
            mae = torch.mean(torch.abs(res - exemplar_gt.squeeze(0)))

            batch_size = exemplar.size(0)
            mae_record.update(mae.item(), batch_size)
            # prediction = np.array(transforms.Resize((h, w))(to_pil(res.cpu())))

            if i > len(val_loader)-3:
                try:
                    save_image(make_grid(torch.cat([examplar_final, exemplar_gt], dim=0), nrow=B), 
                               os.path.join(vis_save_path, f'test_{epoch}_{i}_m.png'))
                    # save_image(make_grid(torch.cat([denorm(exemplar.cpu()), exemplar_ref.cpu()], dim=0), nrow=B), 
                    #            os.path.join(vis_save_path, f'test_{epoch}_{i}.png'))
                except:
                    print('Visualization test error !!')

        log = "val: iter: %d, mae: %f5" % (epoch, mae_record.avg)
        print(log)
        open(val_log_path, 'a').write(log + '\n')

        return mae_record.avg

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='VGD_baseline', help='exp name')
    parser.add_argument('--model', type=str, default='VGD_baseline', help='model name')
    parser.add_argument('--resume', type=str, default=None, help='model name')
    parser.add_argument('--gpu', type=str, default='0', help='used gpu id')
    parser.add_argument('--batchsize', type=int, default=2, help='train batch')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers to load data')
    parser.add_argument('--scale', type=int, default=416)
    parser.add_argument('--bestonly', action="store_true")
    parser.add_argument('--loss_ref_penalty', type=float, default=1)
    parser.add_argument('--finetune_lr', type=float, default=5e-5)
    parser.add_argument('--scratch_lr', type=float, default=5e-4)

    cmd_args = parser.parse_args()

    main(cmd_args)