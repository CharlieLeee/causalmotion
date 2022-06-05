import logging
import os

import torch
import numpy as np

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

from models import CausalMotionModel
from parser import get_evaluation_parser
from loader import data_loader
from utils import *
from loguru import logger

def evaluate(args, loaders, model):
    with torch.no_grad():
        model.eval()
        ade_tot_meter, fde_tot_meter = AverageMeter("ADE", ":.4f"), AverageMeter("FDE", ":.4f")
        for loader in loaders:
            ade_arr = []
            fde_arr = []
            #ade_meter, fde_meter = AverageMeter("ADE", ":.4f"), AverageMeter("FDE", ":.4f")
            for bidx, batch in enumerate(loader):
                batch = [tensor.cuda() for tensor in batch]
                (obs_traj, fut_traj, _, _, _, _, seq_start_end) = batch

                step = args.resume.split('/')[3]
                model_info_tmp = args.resume.split('/')[5]
                epoch = model_info_tmp.split('_')[15]
                # print('model_info_tmp: ', model_info_tmp)
                # print('epoch: ', epoch)
                low_dim = None
                if args.gt_style:
                    print('Using gt_style...')
                    # parse args.filter_envs. e.g. 0.7l
                    # clockwise (l) is 1 ccw (r) is -1
                    if 'r' in args.filter_envs:
                        rule = -1.
                        radius = float(args.filter_envs.replace('r', ''))
                    if 'l' in args.filter_envs:
                        rule = 1. 
                        radius = float(args.filter_envs.replace('l', ''))
                    style_embed = torch.tensor([radius, rule]).cuda()
                    pred_fut_traj_rel, [latent_content_space, first_concat, second_concat] = model(batch, 'P4', style_embed, inspect=True)
                else:
                    print('Not using gt_style...')
                    if args.visualize_embedding:
                        pred_fut_traj_rel, low_dim, [latent_content_space, first_concat, second_concat] = model(batch, 'P6', inspect=True)
                        print('low_dim: ', low_dim.shape)
                    else:
                        pred_fut_traj_rel, [latent_content_space, first_concat, second_concat] = model(batch, step, inspect=True)
                if args.visualize_embedding:
                    # model_info = args.resume.split('/')[5]
                    print("Saving embedding of batch idx: ", bidx)
                    foldername = args.exp + '/' + step + '_' + epoch + args.dset_type 
                    save_filename = str(bidx) + args.filter_envs
                    
                    # same as in the decoder
                    latent_content_space = torch.stack(latent_content_space.split(2, dim=0), dim=0)
                    latent_content_space = latent_content_space.flatten(start_dim=1)
                    
                    embed_dict = {
                        'style_embedding' : low_dim.cpu().detach().numpy() if low_dim is not None else None,
                        'label' : np.array([args.filter_envs]),
                        'latent_content_space': latent_content_space.cpu().detach().numpy(),
                        'first_concat': first_concat.cpu().detach().numpy(),
                        'second_concat': second_concat.cpu().detach().numpy() if second_concat is not None else None
                    }
                    if not os.path.exists('eval_embedding/' + foldername):
                        os.makedirs('eval_embedding/' + foldername)
                    np.save('eval_embedding/' + foldername + '/{}.npy'.format(save_filename), embed_dict)
                   
                pred_fut_traj = relative_to_abs(pred_fut_traj_rel, obs_traj[-1, :, :2]) # [12, 2*bz, 2]
                
                ade_, fde_ = cal_ade_fde(fut_traj, pred_fut_traj)
                ade_, fde_ = ade_ / (obs_traj.shape[1] * fut_traj.shape[0]), fde_ / (obs_traj.shape[1])
                
                # calculate raw ade and fde for visualization
                raw_ade = cal_ade_fde(fut_traj, pred_fut_traj, mode='sumraw')[0][1].cpu().detach().numpy() / fut_traj.shape[0]
                raw_fde = cal_ade_fde(fut_traj, pred_fut_traj, mode='sumraw')[1][1].cpu().detach().numpy()
                
                ade_arr.append(raw_ade)
                fde_arr.append(raw_fde)
                #ade_meter.update(ade_, obs_traj.shape[1]), fde_meter.update(fde_, obs_traj.shape[1])
                ade_tot_meter.update(ade_, obs_traj.shape[1]), fde_tot_meter.update(fde_, obs_traj.shape[1])
                
                if args.visualize_prediction and (bidx == 0):
                    max_id = np.argmax(raw_ade)
                    seq_id = max_id
                    idx_start, idx_end = seq_start_end[seq_id//2][0], seq_start_end[seq_id//2][1]
                    obsv_scene = obs_traj[:, idx_start:idx_end, :]
                    pred_scene = pred_fut_traj[:, idx_start:idx_end, :]
                    gt_scene = fut_traj[:, idx_start:idx_end, :]
                    # compute ADE and FDE metrics
                    if not os.path.exists('./images/eval/visualization/{}_epoch_{}/seed_{}_{}'.format(args.exp, epoch, args.seed, args.dset_type)):
                        os.makedirs('./images/eval/visualization/{}_epoch_{}/seed_{}_{}'.format(args.exp, epoch, args.seed, args.dset_type))
                    figname = './images/eval/visualization/{}_epoch_{}/seed_{}_{}/evalvis_{}_batch_{}_seq_{:02d}_sample_ade{:.3f}_fde{:.3f}.png'.format(
                        args.exp, epoch, args.seed, args.dset_type, args.filter_envs, bidx, seq_id, raw_ade[seq_id], raw_fde[seq_id])
                    figtitle = 'Env {} Batch {} Seq{} ADE{:.3f}  FDE{:.3f}'.format(args.filter_envs, bidx, seq_id, raw_ade[seq_id], raw_fde[seq_id])
                    sceneplot(obsv_scene.permute(1, 0, 2).cpu().detach().numpy(), pred_scene.permute(1, 0, 2).cpu().detach().numpy(), 
                                gt_scene.permute(1, 0, 2).cpu().detach().numpy(), figname=figname, title=figtitle)\
            
            if args.visualize_prediction:
                bar_figname = './images/eval/visualization/{}_epoch_{}/seed_{}_{}/env_{}_mean_ade{:.3f}_fde{:.3f}_bar.png'.format(
                            args.exp, epoch, args.seed, args.dset_type, args.filter_envs, ade_, fde_)
                plotbar(np.concatenate(ade_arr), np.concatenate(fde_arr), figname=bar_figname, title='env_{} ade {:.3f} fde {:.3f}'.format(args.filter_envs, ade_, fde_))

        logging.info('ADE: {:.4f}\tFDE: {:.4f}'.format(ade_tot_meter.avg, fde_tot_meter.avg))


def visualize(args, loader, generator):
    """
    Viasualize some scenes
    """
    keywords = args.resume.split('_')
    suffix = 'ds_' + args.domain_shifts + '_' + keywords[1] + '_irm_' + keywords[3] + '.png'

    # range of idx for visualization
    lb_idx = 44
    ub_idx = 44

    with torch.no_grad():
        for b, data in enumerate(loader):
            batch = [tensor.cuda() for tensor in data]
            (
                obs_traj,
                fut_traj,
                obs_traj_rel,
                _,
                seq_start_end,
                _, _
            ) = batch

            for k in range(args.best_k):
                pred_fut_traj_rel = generator(
                    obs_traj_rel,
                    seq_start_end,
                    0,  # No Teacher
                    3
                )
                pred_fut_traj = relative_to_abs(pred_fut_traj_rel, obs_traj[-1, :, :2])
                idx_sample = seq_start_end.shape[0]
                for i in range(idx_sample):
                    if i < lb_idx or i > ub_idx:
                        continue  # key scenes
                    idx_start, idx_end = seq_start_end[i][0], seq_start_end[i][1]
                    obsv_scene = obs_traj[:, idx_start:idx_end, :]
                    pred_scene = pred_fut_traj[:, idx_start:idx_end, :]
                    gt_scene = fut_traj[:, idx_start:idx_end, :]

                    figname = 'images/visualization/scene_{:02d}_{:02d}_sample_{:02d}_{}'.format(i, b, k, suffix)
                    sceneplot(obsv_scene.permute(1, 0, 2).cpu().detach().numpy(),
                              pred_scene.permute(1, 0, 2).cpu().detach().numpy(),
                              gt_scene.permute(1, 0, 2).cpu().detach().numpy(), figname)


def compute_col(predicted_traj, predicted_trajs_all, thres=0.2, num_interp=4):
    """
    Compute the collisions
    """
    dense_all = interpolate_traj(predicted_trajs_all, num_interp)
    dense_ego = interpolate_traj(predicted_traj[None, :], num_interp)
    distances = np.linalg.norm(dense_all - dense_ego, axis=-1)
    mask = distances[:, 0] > 0
    return distances[mask].min(axis=0) < thres


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    generator = CausalMotionModel(args).cuda()
    load_all_model(args, generator, None)
    envs_path, envs_name = get_envs_path(args.dataset_name, args.dset_type, args.filter_envs)
    loaders = [data_loader(args, env_path, env_name) for env_path, env_name in zip(envs_path, envs_name)]
    logging.info('Model: {}'.format(args.resume))
    logging.info('Split: {}'.format(args.dset_type))
    logging.info('Envs: {}'.format(args.filter_envs))
    logging.info('Seed: {}'.format(args.seed))



    # quantitative
    if args.metrics == 'accuracy':
        evaluate(args, loaders, generator)
        

    # qualitative
    if args.metrics == 'qualitative':
        for loader in loaders:
            visualize(args, loader, generator)

    # collisions [to be implemented]
    if args.metrics == 'collisions':
        for loader in loaders:
            visualize(args, loader, generator)


if __name__ == "__main__":
    args = get_evaluation_parser().parse_args()
    model_param = args.resume.split('/')
    try:
        set_logger(os.path.join(args.log_dir, args.dataset_name,'finetune' if args.finetune else 'pretrain',
                                f'exp_{model_param[2]}_irm_{model_param[3]}_data_{args.dset_type}_{args.filter_envs}_ft_{model_param[4]}_red_{model_param[5][7:-8]}_seed_{args.seed}_reduce_{args.reduce}.log'))
    except:
        pass
    set_seed_globally(args.seed)
    main(args)
